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
            }
        }

    def upscale(self, upscale_model, image, total_megapixels, rescale_method):
        """
        Performs the upscaling or downscaling.

        Args:
            upscale_model: The upscaling model to use.
            image (torch.Tensor): The input image tensor.
            total_megapixels (float): The target total megapixels.
            rescale_method (str): The resampling method for downscaling.

        Returns:
            (torch.Tensor,): A tuple containing the rescaled image tensor.
        """
        # Convert image to comfy's internal format (BCWH)
        samples = image.movedim(-1, 1)
        width = samples.shape[3]
        height = samples.shape[2]

        # Calculate target pixel count
        target_pixels = total_megapixels * 1000000
        current_pixels = width * height

        # --- Step 1: Upscale with Model (if needed) ---
        # If the current image is smaller than the target, use the model to upscale it.
        # This is done first to provide the model with the original image data.
        if current_pixels < target_pixels:
            samples = self.__imageScaler.upscale(upscale_model, image)[0].movedim(-1, 1)

        # --- Step 2: Downscale to Target (if needed) ---
        # After upscaling, the image might be larger than the target.
        # Or, the original image might have been larger than the target.
        # In either case, downscale it to the precise target pixel count.
        current_pixels = samples.shape[3] * samples.shape[2]
        if current_pixels > target_pixels:
            ratio = (target_pixels / current_pixels) ** 0.5
            target_width = round(samples.shape[3] * ratio)
            target_height = round(samples.shape[2] * ratio)
            samples = comfy.utils.common_upscale(samples, target_width, target_height, rescale_method, "disabled")

        # Convert back to standard image format (BHWC)
        samples = samples.movedim(1, -1)
        return (samples,)
