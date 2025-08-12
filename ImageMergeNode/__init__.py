"""
@author: Jules
@title: Image Merge Node
@nickname: Image Merge (Align & Blend)
@description: A node to merge two images with optional alignment and various blending modes.
"""

from .nodes import ImageMergeNode

NODE_CLASS_MAPPINGS = {
    "ImageMergeNode": ImageMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMergeNode": "Image Merge (Align & Blend)"
}
