"""
@author: ominiman
@title: Upscale Image To Total Pixels
@nickname: Upscale Image To Total Pixels
@description: This custom node allow upscaling an image to a desired total pixel count.
"""

from .nodes import UpscaleImageToTotalPixels

NODE_CLASS_MAPPINGS = {
    "UpscaleImageToTotalPixels": UpscaleImageToTotalPixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleImageToTotalPixels": "Upscale Image To Total Pixels"
}
