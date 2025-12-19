import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Jules.PerspectiveWarpNode",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "PerspectiveWarpNode") return;

    // Helper functions for coordinate transformation
    const getDistance = (p1, p2) => Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);

    const canvasToImageCoordinates = (node, canvasX, canvasY) => {
        if (!node.previewImage || !node.image_bounding) return [canvasX, canvasY];
        const [x, y, w, h] = node.image_bounding;
        const scale = w / node.previewImage.width;
        return [(canvasX - x) / scale, (canvasY - y) / scale];
    };

    const imageToCanvasCoordinates = (node, imageX, imageY) => {
        if (!node.previewImage || !node.image_bounding) return [imageX, imageY];
        const [x, y, w, h] = node.image_bounding;
        const scale = w / node.previewImage.width;
        return [(imageX * scale) + x, (imageY * scale) + y];
    };

    // Node lifecycle and interaction logic
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        this.jsonWidget = this.addWidget("string", "points_json", "[]", null, { hidden: true });
        this.addWidget("button", "Reset", null, () => {
            this.properties.points = [];
            this.jsonWidget.value = "[]";
            this.setDirtyCanvas(true);
        });

        this.properties = this.properties || {};
        this.properties.points = [];
        this.previewImage = null;
        this.dragging_point_index = null;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
        onExecuted?.apply(this, arguments);
        if (message?.images) {
            const imageInfo = message.images[0];
            const imageUrl = app.api.apiURL(
                `/view?filename=${encodeURIComponent(imageInfo.filename)}&type=${imageInfo.type}&subfolder=${encodeURIComponent(imageInfo.subfolder)}`
            );
            const img = new Image();
            img.src = imageUrl;
            img.onload = () => {
                this.previewImage = img;
                this.setDirtyCanvas(true, true);
            };
        }
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function(ctx) {
        onDrawForeground?.apply(this, arguments);
        if (!this.previewImage) {
            ctx.font = "bold 16px Arial";
            ctx.fillStyle = "rgba(255, 100, 100, 0.9)";
            ctx.textAlign = "center";
            ctx.fillText("Run upstream node to get preview", this.size[0] / 2, 20);
            this.image_bounding = null;
            return;
        }

        const widget_height = this.widgets.length * 26;
        const available_h = this.size[1] - widget_height;
        const scale = Math.min(this.size[0] / this.previewImage.width, available_h / this.previewImage.height);
        const w = this.previewImage.width * scale;
        const h = this.previewImage.height * scale;
        const x = (this.size[0] - w) / 2;
        const y = (available_h - h) / 2;

        this.image_bounding = [x, y, w, h];
        ctx.drawImage(this.previewImage, x, y, w, h);

        const points = this.properties.points;
        const point_labels = ["1: TL", "2: TR", "3: BL", "4: BR"];
        if (points.length < 4) {
            ctx.font = "bold 16px Arial";
            ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
            ctx.textAlign = "center";
            ctx.fillText(`Click to place ${point_labels[points.length]} corner`, this.size[0] / 2, y > 20 ? y - 5 : 20);
        }

        for (let i = 0; i < points.length; i++) {
            const [cx, cy] = imageToCanvasCoordinates(this, points[i][0], points[i][1]);
            ctx.fillStyle = "rgba(255, 200, 200, 0.9)";
            ctx.beginPath();
            ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillText(point_labels[i], cx + 10, cy + 5);
        }

        if (points.length === 4) {
            ctx.strokeStyle = "rgba(200, 255, 200, 0.9)";
            ctx.lineWidth = 2;
            const p = points.map(pt => imageToCanvasCoordinates(this, pt[0], pt[1]));
            ctx.beginPath();
            ctx.moveTo(p[0][0], p[0][1]);
            ctx.lineTo(p[1][0], p[1][1]);
            ctx.lineTo(p[3][0], p[3][1]);
            ctx.lineTo(p[2][0], p[2][1]);
            ctx.closePath();
            ctx.stroke();
        }
    };

    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function(e) {
        onMouseDown?.apply(this, arguments);
        if (!this.previewImage) return;

        const points = this.properties.points;
        for (let i = 0; i < points.length; i++) {
            const canvasPoint = imageToCanvasCoordinates(this, points[i][0], points[i][1]);
            if (getDistance([e.canvasX, e.canvasY], canvasPoint) < 10) {
                this.dragging_point_index = i;
                return;
            }
        }

        if (points.length < 4) {
            points.push(canvasToImageCoordinates(this, e.canvasX, e.canvasY));
            this.jsonWidget.value = JSON.stringify(points);
        }

        this.setDirtyCanvas(true);
    };

    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function(e) {
        onMouseMove?.apply(this, arguments);
        if (this.dragging_point_index === null) return;

        this.properties.points[this.dragging_point_index] = canvasToImageCoordinates(this, e.canvasX, e.canvasY);
        this.jsonWidget.value = JSON.stringify(this.properties.points);
        this.setDirtyCanvas(true);
    };

    const onMouseUp = nodeType.prototype.onMouseUp;
    nodeType.prototype.onMouseUp = function(e) {
        onMouseUp?.apply(this, arguments);
        this.dragging_point_index = null;
    };
  },
});
