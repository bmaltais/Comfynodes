# ComfyUI/custom_nodes/ImageAspectRatioString/__init__.py

from .nodes import ImageAspectRatioString

NODE_CLASS_MAPPINGS = {
    "ImageAspectRatioString": ImageAspectRatioString
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAspectRatioString": "🖼️ Image Aspect Ratio to String"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-ImageAspectRatioString ###")
