import torch

class ImageAspectRatioString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio_string",)
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "Image"
    OUTPUT_NODE = False

    def get_aspect_ratio(self, image: torch.Tensor):
        batch_size, height, width, num_channels = image.shape
        aspect_ratio = width / height
        return (f"{width}:{height} ({aspect_ratio:.2f})",)
