# ComfyUI/custom_nodes/HelloWorldNode/__init__.py

from .nodes import HelloWorld

NODE_CLASS_MAPPINGS = {
    "HelloWorld": HelloWorld
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HelloWorld": "Hello World Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-HelloWorldNode ###")
