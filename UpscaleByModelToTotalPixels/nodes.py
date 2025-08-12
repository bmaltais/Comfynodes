import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

class UpscaleImageToTotalPixels:
    """
    Upscales an image to a target total pixel count using an upscaling model.
    If the image is already larger than the target, it's downscaled.
    """
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
     
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def __init__(self):
        self.__imageScaler = ImageUpscaleWithModel()
    
    @classmethod
    def INPUT_TYPES(self):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "total_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "rescale_method": (self.rescale_methods,),
                "skip_model_upscale": ("BOOLEAN", {"default": False}),
                "make_divisible_by": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            }
        }

    def upscale(self, upscale_model, image, total_megapixels, rescale_method, skip_model_upscale, make_divisible_by):
        """
        Performs the upscaling or downscaling with optional divisibility constraints.

        Args:
            upscale_model: The upscaling model to use.
            image (torch.Tensor): The input image tensor.
            total_megapixels (float): The target total megapixels.
            rescale_method (str): The resampling method for scaling.
            skip_model_upscale (bool): If True, skips model-based upscaling.
            make_divisible_by (int): Ensures final dimensions are divisible by this number.

        Returns:
            (torch.Tensor,): A tuple containing the rescaled image tensor.
        """
        samples = image.movedim(-1, 1)
        original_width = samples.shape[3]
        original_height = samples.shape[2]

        target_pixels = total_megapixels * 1000000
        current_pixels = original_width * original_height

        # --- Step 1: Initial Upscale (if necessary) ---
        if current_pixels < target_pixels:
            if not skip_model_upscale:
                samples = self.__imageScaler.upscale(upscale_model, image)[0].movedim(-1, 1)
            else:
                # Scale up to the target pixel count using standard resampling
                ratio = (target_pixels / current_pixels) ** 0.5
                target_width = round(original_width * ratio)
                target_height = round(original_height * ratio)
                samples = comfy.utils.common_upscale(samples, target_width, target_height, rescale_method, "disabled")

        # --- Step 2: Calculate Final Dimensions with Divisibility ---
        current_width = samples.shape[3]
        current_height = samples.shape[2]

        # Determine the dimensions needed to hit the target pixel count while maintaining aspect ratio
        ratio = (target_pixels / (current_width * current_height)) ** 0.5
        adjustable_width = round(current_width * ratio)
        adjustable_height = round(current_height * ratio)

        final_width = adjustable_width
        final_height = adjustable_height

        m = make_divisible_by
        if m > 1:
            w, h = adjustable_width, adjustable_height

            def ceil_m(val, mult):
                return (val + mult - 1) // mult * mult

            def floor_m(val, mult):
                return (val // mult) * mult

            w_rem = w % m
            h_rem = h % m

            if not (w_rem == 0 and h_rem == 0):
                # Candidate 1: one dimension up, one down
                if w_rem > h_rem or (w_rem == h_rem and w >= h):
                    cand_w = ceil_m(w, m)
                    cand_h = floor_m(h, m)
                else:
                    cand_h = ceil_m(h, m)
                    cand_w = floor_m(w, m)

                # Check if candidate 1 meets the minimum pixel requirement
                if cand_w * cand_h >= target_pixels:
                    final_width = cand_w
                    final_height = cand_h
                else:
                    # Candidate 2: both dimensions up
                    final_width = ceil_m(w, m)
                    final_height = ceil_m(h, m)

        # --- Step 3: Final Resize ---
        if final_width != current_width or final_height != current_height:
            samples = comfy.utils.common_upscale(samples, final_width, final_height, rescale_method, "disabled")

        samples = samples.movedim(1, -1)
        return (samples,)
