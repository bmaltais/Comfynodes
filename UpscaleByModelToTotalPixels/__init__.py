"""
@author: ominiman
@title: Upscale Image to Total Pixels (using Model)
@nickname: Upscale Image to Total Pixels (using Model)
@description: This node upscales an image to a desired total pixel count using a specified model.
"""

from .nodes import UpscaleImageToTotalPixels

NODE_CLASS_MAPPINGS = {
    "UpscaleImageToTotalPixels": UpscaleImageToTotalPixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleImageToTotalPixels": "Upscale Image to Total Pixels (using Model)"
}
