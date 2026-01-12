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

        # Generate noise map using torch. A single channel for monochrome, or one per channel for color.
        noise_channels = 1 if monochrome else num_channels
        noise_map_small = torch.randn(
            (batch_size, noise_height, noise_width, noise_channels),
            device=image.device,
            dtype=image.dtype
        )

        # Upscale noise to match image dimensions using nearest-neighbor interpolation.
        # This is moved to the GPU to avoid CPU-GPU data transfers.
        # The tensor is permuted to match the (N, C, H, W) format required by interpolate.
        noise_map_small = noise_map_small.permute(0, 3, 1, 2)
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )
        # Permute back to (N, H, W, C) to match the image tensor format.
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # If monochrome, replicate the single noise channel across all image channels.
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, 1, 1, num_channels)

        # Calibrate and apply noise. The mean is subtracted to center the noise distribution,
        # which is then scaled by the intensity. This is a vectorized operation.
        calibrated_noise = (noise_map_full - noise_map_full.mean()) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
