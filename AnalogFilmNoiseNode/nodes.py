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
        Adds film grain to the input image.

        Args:
            image (torch.Tensor): The input image tensor.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain. Larger values create coarser grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        # A larger grain_size results in lower-resolution noise, which is then upscaled.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # --- Start of Vectorized PyTorch Implementation ---
        # This implementation avoids looping over the batch and keeps all operations on the GPU
        # for a significant performance improvement.

        device = image.device
        dtype = image.dtype

        # Determine the shape for the noise tensor.
        noise_channels = 1 if monochrome else num_channels
        noise_shape = (batch_size, noise_height, noise_width, noise_channels)

        # Generate the noise tensor directly on the target device.
        # Using torch.randn is equivalent to np.random.normal for generating normally distributed noise.
        noise_map_small = torch.randn(noise_shape, dtype=dtype, device=device)

        # Upscale the noise map to the original image dimensions using hardware-accelerated interpolation.
        # The input for `interpolate` needs to be in BCHW format, so we permute the dimensions.
        noise_map_small_bchw = noise_map_small.permute(0, 3, 1, 2)
        noise_map_resized_bchw = torch.nn.functional.interpolate(
            noise_map_small_bchw,
            size=(original_height, original_width),
            mode='nearest'  # 'nearest' preserves the blocky appearance of the grain.
        )
        # Permute back to the original BHWC format used by ComfyUI.
        noise_map_full = noise_map_resized_bchw.permute(0, 2, 3, 1)

        # If monochrome, replicate the single noise channel across all image channels.
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, 1, 1, num_channels)

        # Calibrate and apply the noise across the entire batch.
        # 1. Center the noise distribution to have a mean of 0.
        # 2. Scale the noise by the intensity factor.
        # We calculate the mean per-image in the batch to ensure consistent noise application.
        mean_per_image = torch.mean(noise_map_full, dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - mean_per_image) * intensity

        # Add the calibrated noise to the original image and clamp the result to the valid [0.0, 1.0] range.
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
