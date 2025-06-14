# Clear GPU Memory Cache Node for ComfyUI

This custom node provides a simple way to clear the GPU memory cache within ComfyUI. This can be useful for freeing up GPU memory and potentially resolving out-of-memory errors.

## Features

- Simple boolean trigger to clear the cache.
- Prints a message to the console indicating whether the cache was cleared or if CUDA was not available.

## Installation

1. Clone or download this repository.
2. Place the `ClearGpuMemoryCacheNode` directory into your `ComfyUI/custom_nodes/` directory.
3. Restart ComfyUI.

## Usage

1. Add the "Clear GPU Memory Cache" node to your workflow.
2. Connect the `clear_cache_trigger` input to a boolean value (e.g., a primitive boolean node).
3. When the `clear_cache_trigger` is set to `True`, the node will attempt to clear the GPU memory cache.

## Notes

- This node relies on `torch.cuda.empty_cache()` to clear the cache.
- Clearing the cache may not always resolve out-of-memory errors, as memory fragmentation can also be a factor.
- Use this node judiciously, as frequent cache clearing may impact performance.
