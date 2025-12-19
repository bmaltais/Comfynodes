import torch
import numpy as np
import cv2
import json

class PerspectiveWarpNode:
    """
    A node to perform perspective warping on an image, similar to Photoshop's "Perspective Warp".
    The user specifies four corner points of a distorted rectangle, and the node returns a
    cropped, perspective-corrected rectangular image.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. This includes the image to be warped and the
        coordinates of the four corners for the perspective transformation, received from the JS GUI
        as a JSON string.
        """
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "points_json": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_image",)
    FUNCTION = "warp_perspective"
    CATEGORY = "Image/Transform"
    OUTPUT_NODE = False

    def warp_perspective(self, image: torch.Tensor, points_json: str):
        """
        Applies perspective warping to the input image based on four specified corner points.

        Args:
            image (torch.Tensor): The input image tensor (batch).
            points_json (str): A JSON string representing a list of four [x, y] coordinates.

        Returns:
            (torch.Tensor,): A tuple containing the warped image tensor.
        """
        try:
            points = json.loads(points_json)
        except (json.JSONDecodeError, TypeError):
            points = []

        if not points or len(points) != 4:
            return (image,)

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

        return (torch.stack(warped_images),)
