# ImageMergeNode for ComfyUI

The `ImageMergeNode` is a custom node for ComfyUI that provides advanced capabilities for merging and composing two images. It is designed for workflows where one image needs to be aligned, corrected, and blended with a reference image.

## Features

- **Global Alignment**: Automatically aligns the updated image to the original image using feature matching. This is useful for registering images that are slightly misaligned.
- **Facial Correction**: Detects facial landmarks in both images and warps the facial features of the updated image to match the original. This allows for precise correction of facial features like eyes, nose, and mouth.
- **Blending Modes**: Includes several common blending modes (`Normal`, `Multiply`, `Screen`, `Overlay`, `Soft Light`, `Color`) to control how the images are combined.
- **Mixing Strength**: Provides a slider to control the opacity of the blend, allowing for fine-tuning of the final composition.

## Usage

1.  **Add the Node**: Add the `ImageMergeNode` to your ComfyUI workflow.
2.  **Connect Inputs**:
    *   `original_image`: The reference image.
    *   `updated_image`: The image to be modified and blended.
    *   `blending_mode`: The blending mode to use.
    *   `mixing_strength`: The opacity of the blend.
    *   `enable_alignment`: Enable to perform global image alignment.
    *   `enable_facial_correction`: Enable to perform facial warping.
3.  **Run the Workflow**: The node will output the merged image.

## Installation

1.  Clone or download this repository.
2.  Place the `ImageMergeNode` directory into your `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI.
