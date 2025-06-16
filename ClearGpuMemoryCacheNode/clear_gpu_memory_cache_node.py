import torch
import gc

class ClearGpuMemoryCache:
    """
    A node to clear the GPU's memory cache, freeing up VRAM.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # This input is a placeholder to ensure the node executes
                # when the workflow reaches this point.
                "any_type": ("*",)
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "clear_cache"
    OUTPUT_NODE = True
    CATEGORY = "Utilities/Memory"

    def clear_cache(self, any_type):
        """
        This function is executed when the node is triggered.
        It clears the CUDA cache to free up GPU memory.
        """
        print("Running ClearGpuMemoryCache: Clearing VRAM cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return (any_type,)

