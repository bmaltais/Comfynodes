# LatentByMegapixelsAndAspectRatio for ComfyUI

The `LatentByMegapixelsAndAspectRatio` node generates an empty latent image with dimensions calculated based on a target megapixel count and a specific aspect ratio.

## Features

-   **Target Megapixels**: The desired total megapixels for the output image.
-   **Aspect Ratio**: The desired aspect ratio for the output image.
-   **Batch Size**: The number of latent images to generate.
-   **Target Multiplier**: A multiplier to calculate target dimensions for other uses.

## Usage

1.  **Add the Node**: Add the `LatentByMegapixelsAndAspectRatio` node to your ComfyUI workflow.
2.  **Connect Inputs**:
    *   `target_megapixels`: The desired total megapixels.
    *   `aspect_ratio_width`: The width component of the aspect ratio.
    *   `aspect_ratio_height`: The height component of the aspect ratio.
    *   `batch_size`: The number of latent images to generate.
    *   `target_multiplier`: A multiplier to calculate target dimensions for other uses.
3.  **Run the Workflow**: The node will output the latent tensor, base width, base height, target width, and target height.

## Installation

1.  Clone or download this repository.
2.  Place the `LatentByMegapixelsAndAspectRatio` directory into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.
