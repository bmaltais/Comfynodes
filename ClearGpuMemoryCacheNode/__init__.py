# ComfyUI/custom_nodes/ClearGpuMemoryCacheNode/__init__.py

from .clear_gpu_memory_cache_node import ClearGpuMemoryCache

NODE_CLASS_MAPPINGS = {
    "ClearGpuMemoryCache": ClearGpuMemoryCache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearGpuMemoryCache": "🧹 Clear GPU Memory Cache"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-ClearGpuMemoryCacheNode ###")
