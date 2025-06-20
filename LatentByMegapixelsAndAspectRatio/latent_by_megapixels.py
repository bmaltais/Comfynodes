import torch
import comfy.model_management
import math

# Assume MAX_RESOLUTION is defined somewhere globally or accessible.
# For now, let's define a placeholder if it's not directly from comfy.
# MAX_RESOLUTION = 8192 # Example placeholder value
try:
    # Attempt to get it from a common location if it exists
    from comfy.samplers import MAX_RESOLUTION
except ImportError:
    try:
        # Alternative common location
        import nodes
        MAX_RESOLUTION = nodes.MAX_RESOLUTION
    except (ImportError, AttributeError):
        # Fallback if not found, this should be configured correctly in a real setup
        MAX_RESOLUTION = 8192
        print(f"Warning: MAX_RESOLUTION not found in comfy.samplers or nodes. Using fallback: {MAX_RESOLUTION}")


class LatentByMegapixelsAndAspectRatio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        # Definition of MAX_RESOLUTION for INPUT_TYPES (as before)
        try:
            from comfy.samplers import MAX_RESOLUTION as INPUT_MAX_RESOLUTION
        except ImportError:
            try:
                import nodes
                INPUT_MAX_RESOLUTION = nodes.MAX_RESOLUTION
            except (ImportError, AttributeError):
                INPUT_MAX_RESOLUTION = 8192

        return {
            "required": {
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.0625, "max": (INPUT_MAX_RESOLUTION*INPUT_MAX_RESOLUTION)/(1024*1024), "step": 0.1, "tooltip": "Total desired megapixels (e.g., 1.0 for a 1MP image like 1024x1024, 0.25 for 512x512)."}),
                "aspect_ratio_width": ("INT", {"default": 1, "min": 1, "max": INPUT_MAX_RESOLUTION, "step": 1, "tooltip": "Width component of the aspect ratio (e.g., 16 for 16:9)."}),
                "aspect_ratio_height": ("INT", {"default": 1, "min": 1, "max": INPUT_MAX_RESOLUTION, "step": 1, "tooltip": "Height component of the aspect ratio (e.g., 9 for 16:9)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "target_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Multiplier for the target width and height. The original latent dimensions remain based on megapixels and aspect ratio."})
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "TARGET_WIDTH", "TARGET_HEIGHT")
    FUNCTION = "generate"
    CATEGORY = "latent"
    DESCRIPTION = "Generates an empty latent image with a specific total megapixel count and aspect ratio."
    OUTPUT_TOOLTIPS = ("The empty latent image batch (based on original megapixels and aspect ratio).", "Original calculated width (pixels).", "Original calculated height (pixels).", "Target width (pixels, original width * multiplier, rounded to 8).", "Target height (pixels, original height * multiplier, rounded to 8).")

    def generate(self, target_megapixels, aspect_ratio_width, aspect_ratio_height, batch_size=1, target_multiplier=1.0):
        # Ensure aspect ratio components are positive
        if aspect_ratio_width <= 0:
            aspect_ratio_width = 1
        if aspect_ratio_height <= 0:
            aspect_ratio_height = 1

        target_total_pixels = target_megapixels * 1024 * 1024

        # Calculate the area of one "aspect ratio unit block"
        # For an aspect ratio W:H, a block of WxH pixels has W*H area.
        # We want to find a scaling factor 's' such that (s*W) * (s*H) = target_total_pixels
        # s^2 * W * H = target_total_pixels
        # s^2 = target_total_pixels / (W * H)
        # s = sqrt(target_total_pixels / (W * H))

        unit_block_area = aspect_ratio_width * aspect_ratio_height
        if unit_block_area == 0: # Should not happen with positive constraints
            unit_block_area = 1

        scaling_factor = math.sqrt(target_total_pixels / unit_block_area)

        # Calculate initial width and height
        initial_width = aspect_ratio_width * scaling_factor
        initial_height = aspect_ratio_height * scaling_factor

        # Ensure width and height are multiples of 8 for latent space
        # Also, ensure they are at least 8 (or a minimum like 16, but 8 is the divisor)
        width = max(8, round(initial_width / 8.0) * 8)
        height = max(8, round(initial_height / 8.0) * 8)

        # Cap dimensions by MAX_RESOLUTION while maintaining aspect ratio
        current_aspect_ratio = width / height if height != 0 else 1.0

        if width > MAX_RESOLUTION:
            width = MAX_RESOLUTION
            height = max(8, round((width / current_aspect_ratio) / 8.0) * 8)

        if height > MAX_RESOLUTION:
            height = MAX_RESOLUTION
            width = max(8, round((height * current_aspect_ratio) / 8.0) * 8)

        # Recalculate width if height adjustment made it too large (due to rounding)
        if width > MAX_RESOLUTION:
            width = MAX_RESOLUTION
            # No need to adjust height again here, as it's already capped or derived.

        # Ensure minimum dimensions (e.g. 16x16 pixels, which is 2x2 in latent)
        # The max(8, ...) above already ensures at least 8.
        # If a higher minimum like 16 is strictly required for the image pixels:
        min_pixel_dim = 16
        if width < min_pixel_dim:
            width = min_pixel_dim
            # Optionally, re-adjust height to maintain aspect ratio if width was forced up
            # height = max(min_pixel_dim, round((width / current_aspect_ratio) / 8.0) * 8)
            # For simplicity, we'll just enforce min dim without complex re-adjustment here
            # as it might conflict with MAX_RESOLUTION or megapixel target significantly.
        if height < min_pixel_dim:
            height = min_pixel_dim
            # Optionally, re-adjust width
            # width = max(min_pixel_dim, round((height * current_aspect_ratio) / 8.0) * 8)

        # Final check for MAX_RESOLUTION after min_pixel_dim enforcement
        width = min(width, MAX_RESOLUTION)
        height = min(height, MAX_RESOLUTION)

        # These 'width' and 'height' are the final base dimensions for latent generation and original output
        # Calculate target dimensions based on the multiplier
        target_raw_width = width * target_multiplier
        target_raw_height = height * target_multiplier

        # min_pixel_dim is defined earlier in the function (e.g., 16)
        # Ensure target dimensions are multiples of 8 and respect min_pixel_dim
        target_width = max(min_pixel_dim, round(target_raw_width / 8.0) * 8)
        target_height = max(min_pixel_dim, round(target_raw_height / 8.0) * 8)
        
        # Cap target dimensions by MAX_RESOLUTION
        target_width = min(target_width, MAX_RESOLUTION)
        target_height = min(target_height, MAX_RESOLUTION)

        actual_pixels = width * height # Based on original width/height for the latent
        actual_megapixels = actual_pixels / (1024*1024)

        print(f"LatentByMegapixels: Requested {target_megapixels:.2f}MP ({aspect_ratio_width}:{aspect_ratio_height}).")
        print(f"LatentByMegapixels: Initial calculated dimensions: {initial_width:.0f}x{initial_height:.0f}.")
        print(f"LatentByMegapixels: Adjusted to multiples of 8 (base for latent): {width}x{height}.")
        print(f"LatentByMegapixels: Final base dimensions after MAX_RESOLUTION ({MAX_RESOLUTION}) and min_dim ({min_pixel_dim}) constraints: {width}x{height}.")
        print(f"LatentByMegapixels: Resulting actual megapixels for latent: {actual_megapixels:.3f}MP.")
        print(f"LatentByMegapixels: Target multiplier: {target_multiplier:.2f}x.")
        print(f"LatentByMegapixels: Calculated raw target dimensions (width*multiplier, height*multiplier): {width * target_multiplier:.0f}x{height * target_multiplier:.0f}.") 
        print(f"LatentByMegapixels: Final target dimensions (multiple of 8, min_dim {min_pixel_dim}, capped at {MAX_RESOLUTION}): {target_width}x{target_height}.")

        # Latent dimensions are pixel dimensions divided by 8 (using original width/height)
        latent_width = width // 8
        latent_height = height // 8

        if latent_width == 0 or latent_height == 0:
            # This case should be avoided by max(8,...) and min_pixel_dim logic
            # but as a fallback, create a minimal latent.
            print(f"Warning: Calculated latent dimensions are zero ({latent_width}x{latent_height}). Fallback to 1x1 latent (64x64 pixels).")
            latent_width = max(1, latent_width) # At least 1x1 latent block (8x8 pixels)
            latent_height = max(1, latent_height) # So image is 8x8 pixels
            # If min_pixel_dim was 16, then latent min dim is 2.
            latent_width = max(min_pixel_dim // 8, latent_width)
            latent_height = max(min_pixel_dim // 8, latent_height)


        latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=self.device)

        return ({"samples": latent, "ui": {"text": f"{width}x{height} ({actual_megapixels:.2f}MP) -> Target: {target_width}x{target_height}"}}, width, height, target_width, target_height)

# This is needed for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": LatentByMegapixelsAndAspectRatio
}

# Optional: A display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": "Latent by Megapixels & Aspect Ratio"
}
print("LatentByMegapixelsAndAspectRatio node definition loaded")
