import comfy
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

class UpscaleImageToTotalPixels:
    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
     
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def __init__(self):
        self.__imageScaler = ImageUpscaleWithModel()
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "total_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 16.0,
                    "step": 0.1,
                }),
                "rescale_method": (self.rescale_methods,),
            }
        }

    def upscale(self, upscale_model, image, total_megapixels, rescale_method):
        samples = image.movedim(-1,1)

        width = round(samples.shape[3])
        height = round(samples.shape[2])

        target_pixels = total_megapixels * 1000000

        current_pixels = width * height

        if current_pixels < target_pixels:
            samples = self.__imageScaler.upscale(upscale_model, image)[0].movedim(-1,1)

        current_pixels = samples.shape[3] * samples.shape[2]

        if current_pixels > target_pixels:
            ratio = (target_pixels / current_pixels) ** 0.5
            target_width = round(samples.shape[3] * ratio)
            target_height = round(samples.shape[2] * ratio)
            samples = comfy.utils.common_upscale(samples, target_width, target_height, rescale_method, "disabled")

        samples = samples.movedim(1,-1)
        return (samples,)
