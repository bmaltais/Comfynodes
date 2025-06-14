# ComfyUI/custom_nodes/AnalogFilmNoiseNode/__init__.py

from .nodes import AnalogFilmNoiseNode

NODE_CLASS_MAPPINGS = {
    "AnalogFilmNoiseNode": AnalogFilmNoiseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnalogFilmNoiseNode": "🎞️ Analog Film Noise"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-AnalogFilmNoiseNode ###")
