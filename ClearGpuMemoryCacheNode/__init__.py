"""
@author: ominiman
@title: Clear GPU Memory Cache
@nickname: Clear GPU Memory Cache
@description: This node provides a button to clear the GPU's memory cache (VRAM).
"""

from .nodes import ClearGpuMemoryCache

NODE_CLASS_MAPPINGS = {
    "ClearGpuMemoryCache": ClearGpuMemoryCache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearGpuMemoryCache": "🧹 Clear GPU Memory Cache"
}
