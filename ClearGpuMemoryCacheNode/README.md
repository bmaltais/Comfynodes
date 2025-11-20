# ClearGpuMemoryCacheNode for ComfyUI

The `ClearGpuMemoryCacheNode` is a custom node for ComfyUI that clears the GPU memory cache. This can be useful for freeing up GPU memory and potentially resolving out-of-memory errors in complex workflows.

## Features

-   **Trigger**: The node is triggered by any input, allowing it to be placed anywhere in a workflow.
-   **Passthrough**: The node passes through any data it receives, so it can be inserted into a workflow without disrupting the data flow.

## Usage

1.  **Add the Node**: Add the `ClearGpuMemoryCacheNode` to your ComfyUI workflow.
2.  **Connect Inputs**: Connect any output to the `any_type` input of the node to trigger it.
3.  **Run the Workflow**: When the workflow reaches the node, it will clear the GPU memory cache and then pass the input data to any connected output.

## Installation

1.  Clone or download this repository.
2.  Place the `ClearGpuMemoryCacheNode` directory into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.
