import { app } from "/scripts/app.js";

function getDistance(p1, p2) {
    return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
}

app.registerExtension({
    name: "Jules.PerspectiveWarpNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PerspectiveWarpNode") {

            // Helper to transform canvas coordinates to image-space coordinates
            const canvasToImageCoordinates = (node, canvasX, canvasY) => {
                if (!node.image || !node.image_bounding) return [canvasX, canvasY];
                const [x, y, w, h] = node.image_bounding;
                const scale = w / node.image.width;
                return [(canvasX - x) / scale, (canvasY - y) / scale];
            };

            // Helper to transform image-space coordinates to canvas coordinates
            const imageToCanvasCoordinates = (node, imageX, imageY) => {
                if (!node.image || !node.image_bounding) return [imageX, imageY];
                const [x, y, w, h] = node.image_bounding;
                const scale = w / node.image.width;
                return [(imageX * scale) + x, (imageY * scale) + y];
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.jsonWidget = this.widgets.find(w => w.name === "points_json");
                this.points = []; // Points are stored in image-space coordinates
                this.dragging_point_index = null;

                this.addWidget("button", "Reset", null, () => {
                    this.points = [];
                    this.jsonWidget.value = "[]";
                    this.setDirtyCanvas(true, true);
                });
            };

            // Override the onDrawForeground to manually draw the input image
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground?.apply(this, arguments);

                // Get the image from the input slot
                this.image = this.getInputData(0);

                if (!this.image) {
                    ctx.font = "bold 16px Arial";
                    ctx.fillStyle = "rgba(255, 100, 100, 0.9)";
                    ctx.textAlign = "center";
                    ctx.fillText("Connect an image to begin", this.size[0] / 2, 20);
                    this.image_bounding = null; // Clear bounding box if no image
                    return r;
                }

                // Calculate the bounding box to fit and center the image
                const canvasWidth = this.size[0];
                const canvasHeight = this.size[1];
                const imgWidth = this.image.width;
                const imgHeight = this.image.height;
                const widget_height = 26 * this.widgets.length;
                const available_h = canvasHeight - widget_height;
                const scale = Math.min(canvasWidth / imgWidth, available_h / imgHeight);
                const scaledWidth = imgWidth * scale;
                const scaledHeight = imgHeight * scale;
                const x = (canvasWidth - scaledWidth) / 2;
                const y = (available_h - scaledHeight) / 2;

                // Store bounding box for coordinate conversion and draw the image
                this.image_bounding = [x, y, scaledWidth, scaledHeight];
                ctx.drawImage(this.image, x, y, scaledWidth, scaledHeight);

                // Now draw the points and UI on top of the image
                const point_labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Left", "4: Bottom-Right"];
                if (this.points.length < 4) {
                    ctx.font = "bold 16px Arial";
                    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                    ctx.textAlign = "center";
                    const instruction = `Click to place ${point_labels[this.points.length].split(': ')[1]} corner`;
                    ctx.fillText(instruction, this.size[0] / 2, y + 20);
                }

                ctx.strokeStyle = "rgba(255, 200, 200, 0.9)";
                ctx.lineWidth = 2;
                ctx.font = "14px Arial";
                ctx.fillStyle = "rgba(255, 200, 200, 0.9)";

                for (let i = 0; i < this.points.length; i++) {
                    const [canvasX, canvasY] = imageToCanvasCoordinates(this, this.points[i][0], this.points[i][1]);
                    ctx.beginPath();
                    ctx.rect(canvasX - 5, canvasY - 5, 10, 10);
                    ctx.stroke();
                    ctx.fillText(point_labels[i], canvasX + 10, canvasY + 5);
                }

                if (this.points.length === 4) {
                    ctx.strokeStyle = "rgba(200, 255, 200, 0.9)";
                    ctx.beginPath();
                    const p1 = imageToCanvasCoordinates(this, this.points[0][0], this.points[0][1]);
                    const p2 = imageToCanvasCoordinates(this, this.points[1][0], this.points[1][1]);
                    const p3 = imageToCanvasCoordinates(this, this.points[2][0], this.points[2][1]);
                    const p4 = imageToCanvasCoordinates(this, this.points[3][0], this.points[3][1]);
                    ctx.moveTo(p1[0], p1[1]);
                    ctx.lineTo(p2[0], p2[1]);
                    ctx.lineTo(p4[0], p4[1]);
                    ctx.lineTo(p3[0], p3[1]);
                    ctx.closePath();
                    ctx.stroke();
                }

                return r;
            };

            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (e) {
                const r = onMouseDown?.apply(this, arguments);
                if (!this.image || !this.image_bounding) return r;

                for (let i = 0; i < this.points.length; i++) {
                    const canvasPoint = imageToCanvasCoordinates(this, this.points[i][0], this.points[i][1]);
                    if (getDistance([e.canvasX, e.canvasY], canvasPoint) < 10) {
                        this.dragging_point_index = i;
                        return r;
                    }
                }

                if (this.points.length < 4) {
                    const imagePoint = canvasToImageCoordinates(this, e.canvasX, e.canvasY);
                    this.points.push(imagePoint);
                    this.jsonWidget.value = JSON.stringify(this.points);
                }

                this.setDirtyCanvas(true, true);
                return r;
            };

            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (e) {
                const r = onMouseMove?.apply(this, arguments);
                if (this.dragging_point_index === null) return r;

                const imagePoint = canvasToImageCoordinates(this, e.canvasX, e.canvasY);
                this.points[this.dragging_point_index] = imagePoint;
                this.jsonWidget.value = JSON.stringify(this.points);
                this.setDirtyCanvas(true, true);

                return r;
            };

            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function (e) {
                const r = onMouseUp?.apply(this, arguments);
                this.dragging_point_index = null;
                return r;
            };
        }
    },
});
