"""
@author: ominiman
@title: Analog Film Noise
@nickname: Analog Film Noise
@description: This node applies analog film-style noise to an image.
"""

from .nodes import AnalogFilmNoiseNode

NODE_CLASS_MAPPINGS = {
    "AnalogFilmNoiseNode": AnalogFilmNoiseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnalogFilmNoiseNode": "🎞️ Analog Film Noise"
}
