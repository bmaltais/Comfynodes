import torch
import numpy as np
import cv2
import json
import os
from PIL import Image
from folder_paths import get_temp_directory

class PerspectiveWarpNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "points_json": ("STRING", {"default": "[]"}),
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp_perspective"
    CATEGORY = "image/transform"

    def warp_perspective(self, image: torch.Tensor, points_json: str = "[]", node_id=None):
        try:
            points = json.loads(points_json)
        except (json.JSONDecodeError, TypeError):
            points = []

        # Pass through the original image if we don't have four points
        if not points or len(points) != 4:
            return (image,)

        # Save the first image of the batch as a temp file for the preview
        original_filename = None
        if image.shape[0] > 0:
            img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            temp_dir = get_temp_directory()
            filename_hash = hash(f"{node_id}_{image.shape[2]}x{image.shape[1]}")
            original_filename = f"perspective_warp_preview_{filename_hash}.png"
            filepath = os.path.join(temp_dir, original_filename)
            os.makedirs(temp_dir, exist_ok=True)
            try:
                pil_image.save(filepath)
            except Exception as e:
                print(f"[PerspectiveWarpNode] Error saving preview image: {e}")
                original_filename = None

        # Unpack points
        top_left_x, top_left_y = points[0]
        top_right_x, top_right_y = points[1]
        bottom_left_x, bottom_left_y = points[2]
        bottom_right_x, bottom_right_y = points[3]

        batch_size, original_height, original_width, num_channels = image.shape
        warped_images = []

        for i in range(batch_size):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

            src_points = np.float32([
                [top_left_x, top_left_y],
                [top_right_x, top_right_y],
                [bottom_left_x, bottom_left_y],
                [bottom_right_x, bottom_right_y]
            ])

            width_top = np.sqrt(((top_right_x - top_left_x) ** 2) + ((top_right_y - top_left_y) ** 2))
            width_bottom = np.sqrt(((bottom_right_x - bottom_left_x) ** 2) + ((bottom_right_y - bottom_left_y) ** 2))
            max_width = int(max(width_top, width_bottom))

            height_left = np.sqrt(((bottom_left_x - top_left_x) ** 2) + ((bottom_left_y - top_left_y) ** 2))
            height_right = np.sqrt(((bottom_right_x - top_right_x) ** 2) + ((bottom_right_y - top_right_y) ** 2))
            max_height = int(max(height_left, height_right))

            if max_width == 0 or max_height == 0:
                return (image,)

            dst_points = np.float32([
                [0, 0],
                [max_width - 1, 0],
                [0, max_height - 1],
                [max_width - 1, max_height - 1]
            ])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_np = cv2.warpPerspective(img_np, matrix, (max_width, max_height))

            warped_tensor = torch.from_numpy(warped_np.astype(np.float32) / 255.0)
            warped_images.append(warped_tensor)

        warped_batch = torch.stack(warped_images)

        return {
            "ui": {
                "images": [{
                    "filename": original_filename,
                    "subfolder": "",
                    "type": "temp"
                }] if original_filename else []
            },
            "result": (warped_batch,),
        }


NODE_CLASS_MAPPINGS = {
    "PerspectiveWarpNode": PerspectiveWarpNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerspectiveWarpNode": "Perspective Warp",
}
