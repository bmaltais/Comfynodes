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
    
try:
    from .LatentByMegapixelsAndAspectRatio import NODE_CLASS_MAPPINGS as LatentByMegapixelsAndAspectRatio_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as LatentByMegapixelsAndAspectRatio_display_mappings
except ImportError:
    print("[WARN] LatentByMegapixelsAndAspectRatio not found. Skipping.")
    LatentByMegapixelsAndAspectRatio_class_mappings = {}
    LatentByMegapixelsAndAspectRatio_display_mappings = {}

# Import mappings from ImageAspectRatioString
try:
    from .ImageAspectRatioString import NODE_CLASS_MAPPINGS as ImageAspectRatioString_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as ImageAspectRatioString_display_mappings
except ImportError:
    print("[WARN] ImageAspectRatioString not found. Skipping.")
    ImageAspectRatioString_class_mappings = {}
    ImageAspectRatioString_display_mappings = {}

# Aggregate mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(AnalogFilmNoise_class_mappings)
NODE_CLASS_MAPPINGS.update(ClearGpuMemoryCache_class_mappings)
NODE_CLASS_MAPPINGS.update(LatentByMegapixelsAndAspectRatio_class_mappings)
NODE_CLASS_MAPPINGS.update(ImageAspectRatioString_class_mappings)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(AnalogFilmNoise_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(ClearGpuMemoryCache_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(LatentByMegapixelsAndAspectRatio_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(ImageAspectRatioString_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
