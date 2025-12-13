import torch
import numpy as np
import torch.nn.functional as F


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
        Adds film grain to the input image using vectorized GPU operations.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C) on the GPU.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain. Larger values create coarser grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # Get image dimensions from the input tensor
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device  # Keep all tensors on the same device

        # Ensure grain_size is positive to avoid division by zero or invalid dimensions
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions. A larger grain_size results in lower-resolution noise.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # ⚡ OPTIMIZATION: Generate noise for the entire batch directly on the GPU using PyTorch.
        # This avoids slow CPU-based looping and NumPy conversions.
        noise_channels = 1 if monochrome else num_channels
        # Shape: (batch_size, noise_height, noise_width, noise_channels)
        noise_map_small = torch.randn(
            (batch_size, noise_height, noise_width, noise_channels),
            device=device
        )

        # ⚡ OPTIMIZATION: Upscale the noise map on the GPU using efficient interpolation.
        # This replaces the slower np.kron and manual resizing loop.
        # torch.nn.functional.interpolate requires (B, C, H, W) format, so we permute dimensions.
        noise_map_small_permuted = noise_map_small.permute(0, 3, 1, 2)
        noise_map_resized = F.interpolate(
            noise_map_small_permuted,
            size=(original_height, original_width),
            mode='nearest'
        )
        # Permute back to (B, H, W, C) format
        noise_map_full = noise_map_resized.permute(0, 2, 3, 1)

        # If monochrome, replicate the single noise channel across all image channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, 1, 1, num_channels)

        # ⚡ OPTIMIZATION: Calibrate and apply noise using vectorized tensor operations.
        # This is significantly faster than processing each image individually.
        # We calculate the mean across spatial and channel dimensions for each item in the batch.
        mean_per_item = noise_map_full.mean(dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - mean_per_item) * intensity

        # Add noise to the original image and clamp the result to the valid [0.0, 1.0] range
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
