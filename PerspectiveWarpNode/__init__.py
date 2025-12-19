from .nodes import PerspectiveWarpNode

NODE_CLASS_MAPPINGS = {
    "PerspectiveWarpNode": PerspectiveWarpNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerspectiveWarpNode": "Perspective Warp"
}

WEB_DIRECTORY = "web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
