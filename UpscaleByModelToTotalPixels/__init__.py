"""
@author: ominiman
@title: Upscale Image To Total Pixels (using Model)
@nickname: Upscale Image To Total Pixels (using Model)
@description: This custom node allow upscaling an image to a desired total pixel count using a specified model. If the input image is larger than the target, it will be downscaled to fit the total pixel count without getting upscaled by the model first.
"""

from .nodes import UpscaleImageToTotalPixels

NODE_CLASS_MAPPINGS = {
    "UpscaleImageToTotalPixels": UpscaleImageToTotalPixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleImageToTotalPixels": "Upscale Image to Total Pixels (using Model)"
}
