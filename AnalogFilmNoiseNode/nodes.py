import torch
import numpy as np

class AnalogFilmNoiseNode:
    """
    Applies analog film-style noise to an image. This effect simulates the grain
    found in traditional photographic film.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node, including the image, noise intensity,
        grain size, and monochrome option.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "monochrome": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_noise",)
    FUNCTION = "apply_film_noise"
    CATEGORY = "Image/Effects"
    OUTPUT_NODE = False

    def apply_film_noise(self, image: torch.Tensor, intensity: float, grain_size: float, monochrome: bool):
        """
        Adds film grain to the input image using a fully vectorized PyTorch implementation.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C).
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain. Larger values create coarser grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # ⚡ Bolt Optimization: This entire function was refactored to use vectorized PyTorch operations.
        # The original implementation looped through each image in the batch, converting it to a NumPy
        # array on the CPU for processing. This caused a significant performance bottleneck due to
        # CPU-GPU data transfers inside the loop. This version keeps all data on the GPU and
        # processes the entire batch at once, resulting in a substantial speedup.

        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions. A larger grain_size results in lower-resolution noise, which is then upscaled.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate the noise tensor directly on the target device (e.g., GPU).
        # The shape is (B, C, H, W) which is the standard PyTorch format for image processing filters.
        if monochrome:
            # For monochrome, create a single-channel noise map.
            noise_channels = 1
        else:
            # For color, create noise for each channel independently.
            noise_channels = num_channels

        noise_shape = (batch_size, noise_channels, noise_height, noise_width)
        noise_small = torch.randn(noise_shape, device=device)

        # Upscale the low-resolution noise map to the full image size using nearest-neighbor interpolation.
        # This is a GPU-accelerated operation that preserves the blocky, grainy appearance.
        noise_resized = torch.nn.functional.interpolate(
            noise_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channel count.
        # .expand() is a memory-efficient operation that doesn't copy data.
        if monochrome:
            noise_resized = noise_resized.expand(-1, num_channels, -1, -1)

        # Permute the noise tensor from PyTorch's standard (B, C, H, W) to match the input image's (B, H, W, C) layout.
        noise_map_full = noise_resized.permute(0, 2, 3, 1)

        # Calibrate and apply the noise across the entire batch at once.
        # 1. Center the noise distribution around 0 by subtracting the mean of each image's noise map.
        #    `view` and `mean` are used to compute the mean per image in the batch.
        # 2. Scale the noise by the intensity factor.
        mean_per_image = torch.mean(noise_map_full.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
        calibrated_noise = (noise_map_full - mean_per_image) * intensity

        # Add the calibrated noise to the original image and clamp the result to the valid [0.0, 1.0] range.
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
