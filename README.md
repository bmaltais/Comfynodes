# ComfyUI-Bolt-Utilities ⚡

A collection of high-performance, valuable utility nodes for ComfyUI.

## Nodes

### 🎨 Color Match (Reinhard)
Adjusts the color profile of a target image to match a reference image using the Reinhard method.
- **Benefit:** Ensures consistent lighting and color across different images, which is essential for seamless compositing, inpainting, and face-swapping.
- **How it works:** It matches the mean and standard deviation of the images in the LAB color space, effectively transferring the "feel" of the reference image to the target.

### 🎞️ Analog Film Noise
Applies realistic analog film-style noise to an image.
- **Benefit:** Adds character and texture to digital images, simulating the organic look of traditional film grain.

### 🧹 Clear GPU Memory Cache
Clears the CUDA cache and performs garbage collection.
- **Benefit:** Helps manage VRAM in complex workflows, preventing "Out of Memory" errors by manually freeing up unused resources.

### 🛰️ Image Merge (Align & Blend)
Merges two images with optional automatic alignment and facial correction.
- **Benefit:** Simplifies the process of blending generated images with original photos, especially when there have been slight shifts in composition or facial features.

### 📐 Latent by Megapixels & Aspect Ratio
Generates an empty latent based on a target megapixel count and aspect ratio.
- **Benefit:** Allows for more intuitive resolution control (e.g., "1MP at 16:9") compared to manually calculating pixel dimensions.

### 🚀 Upscale Image to Total Pixels
Upscales an image to a target total pixel count using a model-based upscaler and smart resampling.
- **Benefit:** Replaces multiple nodes with a single, efficient operation that guarantees a specific output resolution while maintaining quality.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
