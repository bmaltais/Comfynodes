print("### Loading: Custom Nodes Collection ###")

# Import mappings from AnalogFilmNoiseNode
try:
    from .AnalogFilmNoiseNode import NODE_CLASS_MAPPINGS as AnalogFilmNoise_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as AnalogFilmNoise_display_mappings
except ImportError:
    print("[WARN] AnalogFilmNoiseNode not found. Skipping.")
    AnalogFilmNoise_class_mappings = {}
    AnalogFilmNoise_display_mappings = {}

# Import mappings from ClearGpuMemoryCacheNode
try:
    from .ClearGpuMemoryCacheNode import NODE_CLASS_MAPPINGS as ClearGpuMemoryCache_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as ClearGpuMemoryCache_display_mappings
except ImportError:
    print("[WARN] ClearGpuMemoryCacheNode not found. Skipping.")
    ClearGpuMemoryCache_class_mappings = {}
    ClearGpuMemoryCache_display_mappings = {}

# Aggregate mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(AnalogFilmNoise_class_mappings)
NODE_CLASS_MAPPINGS.update(ClearGpuMemoryCache_class_mappings)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(AnalogFilmNoise_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(ClearGpuMemoryCache_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
