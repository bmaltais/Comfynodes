# custom_nodes/__init__.py

# Import all node modules from this directory
from .latent_by_megapixels import NODE_CLASS_MAPPINGS as lbm_mappings
from .latent_by_megapixels import NODE_DISPLAY_NAME_MAPPINGS as lbm_display_mappings

# Combine all mappings
NODE_CLASS_MAPPINGS = {**lbm_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**lbm_display_mappings}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Custom nodes initialized from custom_nodes directory")
