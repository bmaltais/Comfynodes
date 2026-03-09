import { app } from "../../scripts/app.js";

// ─── Constants ───────────────────────────────────────────────────────────────

const NODE_NAME         = "PerspectiveCorrectionNode";
const WIDGET_ROW_H      = 25;   // pixels per visible widget row (matches LiteGraph default)
const PREVIEW_PAD_X     = 10;
const PREVIEW_PAD_TOP   = 6;
const BOTTOM_MARGIN     = 22;   // space below preview for instruction text
const DEFAULT_W         = 380;
const DEFAULT_PREVIEW_H = 280;
const HANDLE_R          = 8;
const HANDLE_R_ACTIVE   = 12;
const HIT_R             = 14;

const POINT_COLORS = ["#FF5555", "#55DD55", "#5599FF", "#FFCC44"];
const QUAD_COLOR   = "#FF8800";

// ─── Extension ───────────────────────────────────────────────────────────────

app.registerExtension({
    name: "comfynodes.perspectivecorrection",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        // ── onNodeCreated ─────────────────────────────────────────────────────
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);

            this.serialize_widgets = true;

            // Per-node interactive state
            this._persp = {
                points:      [],   // [{x,y}] normalised [0..1] within image
                draggingIdx: -1,
                hoverIdx:    -1,
                imgLoaded:   false,
                img:         null,
            };

            // Hide the corner_points STRING widget (keeps its value but takes no space)
            const cpw = this.widgets?.find(w => w.name === "corner_points");
            if (cpw) {
                cpw.type        = "hidden";
                cpw.hidden      = true;        // belt-and-suspenders
                cpw.computeSize = () => [0, -4];
            }

            // Visible Reset button
            this.addWidget("button", "Reset Points", null, () => {
                this._persp.points = [];
                _syncWidget(this);
                this.setDirtyCanvas(true);
            });

            // Set instance-level mouse handlers here so LiteGraph finds them reliably
            _bindMouseHandlers(this);

            const wh = _widgetAreaH(this);
            this.size = [DEFAULT_W, wh + PREVIEW_PAD_TOP + DEFAULT_PREVIEW_H + BOTTOM_MARGIN];
        };

        // ── onAdded ───────────────────────────────────────────────────────────
        // Re-bind in onAdded too (called after workflow load), to be safe.
        const origAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            origAdded?.apply(this, arguments);
            _bindMouseHandlers(this);
        };

        // ── onExecuted ───────────────────────────────────────────────────────
        nodeType.prototype.onExecuted = function (message) {
            if (!message?.images?.length) return;

            const { filename, type, subfolder } = message.images[0];
            const url =
                `/view?filename=${encodeURIComponent(filename)}` +
                `&type=${encodeURIComponent(type)}` +
                `&subfolder=${encodeURIComponent(subfolder ?? "")}` +
                `&rand=${Math.random()}`;

            const img = new Image();
            img.onload = () => {
                this._persp.img       = img;
                this._persp.imgLoaded = true;

                // Resize the node so the preview matches the image aspect ratio
                const nodeW    = this.size[0];
                const previewW = nodeW - PREVIEW_PAD_X * 2;
                const previewH = Math.round(previewW * img.naturalHeight / img.naturalWidth);
                const wh       = _widgetAreaH(this);
                this.size[1]   = wh + PREVIEW_PAD_TOP + Math.max(previewH, 100) + BOTTOM_MARGIN;

                this.setDirtyCanvas(true);
            };
            img.onerror = () => console.warn("[PerspCorrection] preview image failed to load");
            img.src = url;
        };

        // ── onDrawForeground ─────────────────────────────────────────────────
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (this.flags?.collapsed) return;
            if (!this._persp) return;

            const state   = this._persp;
            const preview = _previewArea(this);

            // 1. Image or placeholder
            if (state.imgLoaded && state.img) {
                ctx.drawImage(state.img, preview.x, preview.y, preview.w, preview.h);
            } else {
                ctx.fillStyle = "#1a1a2e";
                ctx.fillRect(preview.x, preview.y, preview.w, preview.h);
                ctx.fillStyle    = "#666";
                ctx.font         = "12px sans-serif";
                ctx.textAlign    = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    "Run the node once to load the image,",
                    preview.x + preview.w / 2, preview.y + preview.h / 2 - 8,
                );
                ctx.fillText(
                    "then click 4 corners to define perspective.",
                    preview.x + preview.w / 2, preview.y + preview.h / 2 + 8,
                );
            }

            // 2. Border
            ctx.strokeStyle = "#555";
            ctx.lineWidth   = 1;
            ctx.strokeRect(preview.x, preview.y, preview.w, preview.h);

            // 3. Quadrilateral overlay
            if (state.points.length >= 2) {
                const cpts = state.points.map(p => _toCanvas(p, preview));
                ctx.save();
                ctx.strokeStyle = QUAD_COLOR;
                ctx.lineWidth   = 2;
                if (state.points.length < 4) ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(cpts[0].x, cpts[0].y);
                for (let i = 1; i < cpts.length; i++) ctx.lineTo(cpts[i].x, cpts[i].y);
                if (state.points.length === 4) ctx.closePath();
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }

            // 4. Corner handles
            for (let i = 0; i < state.points.length; i++) {
                const cp     = _toCanvas(state.points[i], preview);
                const active = state.hoverIdx === i || state.draggingIdx === i;
                const r      = active ? HANDLE_R_ACTIVE : HANDLE_R;

                ctx.save();
                ctx.shadowColor = "rgba(0,0,0,0.6)";
                ctx.shadowBlur  = 6;
                ctx.beginPath();
                ctx.arc(cp.x, cp.y, r, 0, Math.PI * 2);
                ctx.fillStyle = POINT_COLORS[i % POINT_COLORS.length];
                ctx.fill();
                ctx.shadowBlur  = 0;
                ctx.strokeStyle = "white";
                ctx.lineWidth   = 1.5;
                ctx.stroke();
                ctx.fillStyle    = "black";
                ctx.font         = `bold ${Math.round(r * 1.1)}px sans-serif`;
                ctx.textAlign    = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(i + 1, cp.x, cp.y);
                ctx.restore();
            }

            // 5. Instruction
            const msg = state.points.length < 4
                ? `Click inside the image to place point ${state.points.length + 1} / 4`
                : "Drag handles to adjust  •  Right-click a handle to remove it";
            ctx.fillStyle    = "#999";
            ctx.font         = "11px sans-serif";
            ctx.textAlign    = "center";
            ctx.textBaseline = "top";
            ctx.fillText(msg, preview.x + preview.w / 2, preview.y + preview.h + 4);
        };

        // ── onConfigure ───────────────────────────────────────────────────────
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            // Ensure state exists (might be called before onNodeCreated in rare cases)
            if (!this._persp) this._persp = { points: [], draggingIdx: -1, hoverIdx: -1, imgLoaded: false, img: null };
            const cpw = this.widgets?.find(w => w.name === "corner_points");
            if (!cpw) return;
            try {
                const raw = JSON.parse(cpw.value || "[]");
                this._persp.points = raw.map(p => ({ x: p[0], y: p[1] }));
            } catch {
                this._persp.points = [];
            }
        };
    },
});

// ─── Per-instance mouse handler binding ──────────────────────────────────────
// Must be instance-level (not prototype) so LiteGraph finds them reliably.
// Uses e.canvasX / e.canvasY (graph-space) minus node.pos, matching how the
// reference DragCrop node handles coordinates.

function _bindMouseHandlers(node) {

    node.onMouseDown = function (e) {
        if (!this._persp) return false;
        const state   = this._persp;
        const preview = _previewArea(this);

        // Convert canvas-space → node-body-space
        const lx = e.canvasX - this.pos[0];
        const ly = e.canvasY - this.pos[1];

        if (!_inPreview(lx, ly, preview)) return false;

        const localX = lx - preview.x;
        const localY = ly - preview.y;

        // Right-click → remove nearest point
        if (e.button === 2) {
            const idx = _nearestPoint(state.points, localX, localY, preview, HIT_R * 2);
            if (idx >= 0) {
                state.points.splice(idx, 1);
                _syncWidget(this);
                this.setDirtyCanvas(true);
                return true;
            }
            return false;
        }

        // Left-click on existing handle → drag
        const hit = _nearestPoint(state.points, localX, localY, preview, HIT_R);
        if (hit >= 0) {
            state.draggingIdx = hit;
            return true;   // capture mouse for drag
        }

        // Left-click on empty space → add point (max 4)
        if (state.points.length < 4) {
            state.points.push(_toNorm(localX, localY, preview));
            _syncWidget(this);
            this.setDirtyCanvas(true);
            return true;
        }

        return false;
    };

    node.onMouseMove = function (e) {
        if (!this._persp) return false;
        const state   = this._persp;
        const preview = _previewArea(this);

        const lx     = e.canvasX - this.pos[0];
        const ly     = e.canvasY - this.pos[1];
        const localX = lx - preview.x;
        const localY = ly - preview.y;

        // Hover highlight
        const hover = _nearestPoint(state.points, localX, localY, preview, HIT_R * 1.4);
        if (hover !== state.hoverIdx) {
            state.hoverIdx = hover;
            this.setDirtyCanvas(true);
        }

        // Drag active point
        if (state.draggingIdx >= 0) {
            const cx = Math.max(0, Math.min(preview.w, localX));
            const cy = Math.max(0, Math.min(preview.h, localY));
            state.points[state.draggingIdx] = _toNorm(cx, cy, preview);
            _syncWidget(this);
            this.setDirtyCanvas(true);
            return true;
        }

        return false;
    };

    node.onMouseUp = function (_e) {
        if (!this._persp) return false;
        const state = this._persp;
        if (state.draggingIdx >= 0) {
            state.draggingIdx = -1;
            this.setDirtyCanvas(true);
            return true;
        }
        return false;
    };

    node.onMouseLeave = function () {
        if (!this._persp) return;
        this._persp.hoverIdx    = -1;
        this._persp.draggingIdx = -1;
        this.setDirtyCanvas(true);
    };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Preview rectangle in node-body-local coordinates (origin = body top-left). */
function _previewArea(node) {
    const wh = _widgetAreaH(node);
    const x  = PREVIEW_PAD_X;
    const y  = wh + PREVIEW_PAD_TOP;
    const w  = Math.max(10, node.size[0] - PREVIEW_PAD_X * 2);
    const h  = Math.max(10, node.size[1] - y - BOTTOM_MARGIN);
    return { x, y, w, h };
}

/** Sum of heights of all visible (non-hidden) widget rows. */
function _widgetAreaH(node) {
    let h = 4;
    for (const w of (node.widgets ?? [])) {
        if (!w) continue;
        if (w.type === "hidden" || w.hidden === true) continue;
        // computeSize may or may not exist; fall back to a sensible constant
        let wh;
        try { wh = w.computeSize?.(node.size?.[0] ?? DEFAULT_W)?.[1]; }
        catch { wh = undefined; }
        if (wh == null || wh <= 0) wh = WIDGET_ROW_H;
        h += wh + 4;
    }
    return h;
}

function _inPreview(lx, ly, p) {
    return lx >= p.x && lx <= p.x + p.w && ly >= p.y && ly <= p.y + p.h;
}

function _toNorm(localX, localY, preview) {
    return { x: localX / preview.w, y: localY / preview.h };
}

function _toCanvas(norm, preview) {
    return { x: preview.x + norm.x * preview.w, y: preview.y + norm.y * preview.h };
}

/** Return index of nearest point within threshold, or -1. */
function _nearestPoint(points, localX, localY, preview, threshold) {
    let bestIdx  = -1;
    let bestDist = threshold;
    for (let i = 0; i < points.length; i++) {
        const d = Math.hypot(
            points[i].x * preview.w - localX,
            points[i].y * preview.h - localY,
        );
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
}

/** Serialise the current points array into the hidden corner_points widget. */
function _syncWidget(node) {
    const w = node.widgets?.find(w => w.name === "corner_points");
    if (w) w.value = JSON.stringify(node._persp.points.map(p => [p.x, p.y]));
}
