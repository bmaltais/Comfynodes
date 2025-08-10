import torch
import gc

class ClearGpuMemoryCache:
    """
    A node to clear the GPU's memory cache, freeing up VRAM. It can be used
    to manage memory in complex workflows.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. It accepts any input as a trigger.
        """
        return {
            "required": {
                # This input is a placeholder to ensure the node executes
                # when the workflow reaches this point. It is passed through unmodified.
                "any_type": ("*",)
            }
        }

    # The node passes through the input it receives, without modification.
    RETURN_TYPES = ("*",)
    FUNCTION = "clear_cache"
    OUTPUT_NODE = True
    CATEGORY = "Utilities/Memory"

    def clear_cache(self, any_type):
        """
        Clears the CUDA cache to free up GPU memory and performs garbage collection.

        Args:
            any_type: Any data type, used as a trigger for execution and passed through.

        Returns:
            (any,): A tuple containing the unmodified input data.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return (any_type,)
