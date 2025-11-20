import torch
import gc

class ClearGpuMemoryCache:
    """
    A node to clear the GPU's memory cache, freeing up VRAM.

    This node is designed to be used in complex workflows where GPU memory management
    is critical. It calls `torch.cuda.empty_cache()` and `gc.collect()` to release
    unreferenced memory back to the system, which can help prevent out-of-memory
    errors.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. It accepts any input as a trigger
        to ensure execution within the workflow.

        Returns:
            dict: A dictionary specifying the required input types for the node.
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

        This method is the primary function of the node. It checks for CUDA availability
        before attempting to clear the cache and also runs Python's garbage collector
        to ensure a thorough cleanup.

        Args:
            any_type: Any data type, used as a trigger for execution and passed through.

        Returns:
            (any,): A tuple containing the unmodified input data, allowing the node to
                    be inserted anywhere in a workflow without disrupting data flow.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return (any_type,)
