import torch
import comfy.model_management
import math

# Attempt to import MAX_RESOLUTION from ComfyUI's samplers, with a fallback for safety.
try:
    from comfy.samplers import MAX_RESOLUTION
except ImportError:
    MAX_RESOLUTION = 8192

class LatentByMegapixelsAndAspectRatio:
    """
    Generates an empty latent image with dimensions calculated based on a target
    megapixel count and a specific aspect ratio.
    """
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.0625, "max": (MAX_RESOLUTION*MAX_RESOLUTION)/(1024*1024), "step": 0.1}),
                "aspect_ratio_width": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "aspect_ratio_height": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "target_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "TARGET_WIDTH", "TARGET_HEIGHT")
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(self, target_megapixels, aspect_ratio_width, aspect_ratio_height, batch_size=1, target_multiplier=1.0):
        """
        Calculates dimensions from megapixels and aspect ratio, then creates an empty latent.

        Args:
            target_megapixels (float): The desired total megapixels (e.g., 1.0 for 1024x1024).
            aspect_ratio_width (int): The width component of the aspect ratio.
            aspect_ratio_height (int): The height component of the aspect ratio.
            batch_size (int): The number of latent images to generate.
            target_multiplier (float): A multiplier to calculate target dimensions for other uses.

        Returns:
            (dict, int, int, int, int): A tuple containing the latent tensor, base width,
            base height, target width, and target height.
        """
        # Calculate total pixels from megapixels
        target_total_pixels = target_megapixels * 1024 * 1024

        # Calculate a scaling factor 's' such that (s * W) * (s * H) = total_pixels
        unit_block_area = aspect_ratio_width * aspect_ratio_height
        scaling_factor = math.sqrt(target_total_pixels / unit_block_area)

        # Calculate initial dimensions
        initial_width = aspect_ratio_width * scaling_factor
        initial_height = aspect_ratio_height * scaling_factor

        # Round to the nearest multiple of 8, ensuring a minimum of 8
        width = max(8, round(initial_width / 8.0) * 8)
        height = max(8, round(initial_height / 8.0) * 8)

        # Enforce MAX_RESOLUTION while attempting to maintain aspect ratio
        current_aspect_ratio = width / height if height != 0 else 1.0
        if width > MAX_RESOLUTION:
            width = MAX_RESOLUTION
            height = max(8, round((width / current_aspect_ratio) / 8.0) * 8)
        if height > MAX_RESOLUTION:
            height = MAX_RESOLUTION
            width = max(8, round((height * current_aspect_ratio) / 8.0) * 8)

        # Final cap to ensure dimensions are within limits after adjustments
        width = min(width, MAX_RESOLUTION)
        height = min(height, MAX_RESOLUTION)

        # A common minimum dimension for stable diffusion is 16
        min_pixel_dim = 16
        width = max(min_pixel_dim, width)
        height = max(min_pixel_dim, height)

        # Calculate target dimensions based on the multiplier
        target_width = max(min_pixel_dim, round((width * target_multiplier) / 8.0) * 8)
        target_height = max(min_pixel_dim, round((height * target_multiplier) / 8.0) * 8)

        # Cap target dimensions by MAX_RESOLUTION
        target_width = min(target_width, MAX_RESOLUTION)
        target_height = min(target_height, MAX_RESOLUTION)

        # Create the empty latent tensor
        latent_width = width // 8
        latent_height = height // 8
        latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=self.device)

        actual_megapixels = (width * height) / (1024*1024)
        ui_text = f"{width}x{height} ({actual_megapixels:.2f}MP) -> Target: {target_width}x{target_height}"

        return ({"samples": latent, "ui": {"text": ui_text}}, width, height, target_width, target_height)

NODE_CLASS_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": LatentByMegapixelsAndAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": "Latent by Megapixels & Aspect Ratio"
}
