# ComfyUI Custom Nodes

This repository is a collection of custom nodes for ComfyUI, designed to extend its capabilities with advanced image and latent manipulation features.

## Nodes

This collection includes the following nodes:

-   **AnalogFilmNoiseNode**: Applies analog film-style noise to an image.
-   **ClearGpuMemoryCacheNode**: Clears the GPU memory cache to free up VRAM.
-   **ImageMergeNode**: Merges two images with optional alignment and various blending modes.
-   **LatentByMegapixelsAndAspectRatio**: Generates an empty latent image with dimensions calculated based on a target megapixel count and a specific aspect ratio.
-   **UpscaleByModelToTotalPixels**: Upscales an image to a target total pixel count using an upscaling model.

## Installation

1.  Clone this repository into `ComfyUI/custom_nodes/`:
    ```bash
    git clone https://github.com/your-username/comfyui-custom-nodes.git path/to/ComfyUI/custom_nodes/comfyui-custom-nodes
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Restart ComfyUI.

## Usage

Each node is available in its own category within the ComfyUI menu. For detailed information on each node, please refer to the `README.md` in the corresponding node's directory.
