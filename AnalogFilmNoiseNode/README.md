# AnalogFilmNoiseNode for ComfyUI

The `AnalogFilmNoiseNode` is a custom node for ComfyUI that applies analog film-style noise to an image. This effect simulates the grain found in traditional photographic film.

## Features

-   **Intensity**: Controls the strength of the noise effect.
-   **Grain Size**: Controls the size of the noise grain. Larger values create coarser grain.
-   **Monochrome**: If `True`, applies grayscale noise; otherwise, applies color noise.

## Usage

1.  **Add the Node**: Add the `AnalogFilmNoiseNode` to your ComfyUI workflow.
2.  **Connect Inputs**:
    *   `image`: The input image.
    *   `intensity`: The strength of the noise effect.
    *   `grain_size`: The size of the noise grain.
    *   `monochrome`: `True` for grayscale noise, `False` for color noise.
3.  **Run the Workflow**: The node will output the image with added film noise.

## Installation

1.  Clone or download this repository.
2.  Place the `AnalogFilmNoiseNode` directory into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.
