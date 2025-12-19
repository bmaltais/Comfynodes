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
        Adds film grain to the input image using a vectorized, GPU-accelerated approach.

        Args:
            image (torch.Tensor): The input image tensor in (B, H, W, C) format.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # Optimization: All operations are performed on the GPU using PyTorch tensors
        # to avoid costly CPU-GPU data transfers and leverage parallel processing.
        # The previous implementation iterated through the batch and used CPU-bound NumPy,
        # which is a significant performance bottleneck.
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        grain_size = max(0.1, grain_size)
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # PyTorch's interpolate function expects (B, C, H, W), so we permute the dimensions.
        image_nchw = image.permute(0, 3, 1, 2)

        # Generate noise tensor directly on the GPU.
        noise_channels = 1 if monochrome else num_channels
        noise_map_small = torch.randn(
            (batch_size, noise_channels, noise_height, noise_width),
            device=device,
            dtype=image.dtype
        )

        # Upscale the noise map using nearest-neighbor interpolation on the GPU.
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channel count.
        if monochrome:
            noise_map_full = noise_map_full.expand(-1, num_channels, -1, -1)

        # Calibrate noise: center it around zero and scale by intensity.
        calibrated_noise = (noise_map_full - noise_map_full.mean()) * intensity

        # Add noise to the image and clip the result to the valid [0.0, 1.0] range.
        noisy_image_nchw = (image_nchw + calibrated_noise).clamp_(0.0, 1.0)

        # Permute the dimensions back to the original (B, H, W, C) format.
        noisy_image = noisy_image_nchw.permute(0, 2, 3, 1)

        return (noisy_image,)
