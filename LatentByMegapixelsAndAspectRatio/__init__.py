"""
@author: ominiman
@title: Latent by Megapixels and Aspect Ratio
@nickname: Latent by Megapixels and Aspect Ratio
@description: This node generates an empty latent image with a specific total megapixel count and aspect ratio.
"""

from .nodes import LatentByMegapixelsAndAspectRatio

NODE_CLASS_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": LatentByMegapixelsAndAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentByMegapixelsAndAspectRatio": "🖼️ Latent By Megapixels And Aspect Ratio"
}
