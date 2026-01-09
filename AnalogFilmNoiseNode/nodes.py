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
        Adds film grain to the input image using a vectorized PyTorch implementation.
        This approach avoids CPU-GPU synchronization and processes the entire batch on the GPU for better performance.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C).
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # Move tensor to the correct device
        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Determine the number of noise channels
        noise_channels = 1 if monochrome else num_channels

        # Generate noise for the entire batch on the GPU
        # ⚡ Bolt Optimization: This is the core optimization. By generating noise for the entire batch at once
        # on the GPU, we avoid the previous implementation's costly loop and CPU-GPU data transfers.
        noise_map_small = torch.randn(
            (batch_size, noise_height, noise_width, noise_channels),
            device=device
        )

        # Upscale noise to match image dimensions using nearest-neighbor interpolation
        # Permute to (B, C, H, W) for interpolate
        noise_map_small_permuted = noise_map_small.permute(0, 3, 1, 2)
        noise_map_resized = torch.nn.functional.interpolate(
            noise_map_small_permuted,
            size=(original_height, original_width),
            mode='nearest'
        )
        # Permute back to (B, H, W, C)
        noise_map_full = noise_map_resized.permute(0, 2, 3, 1)

        # If monochrome, expand the single noise channel to match the image's channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.expand(-1, -1, -1, num_channels)

        # Calibrate and apply noise across the batch
        # Center the noise distribution and scale by intensity
        # We calculate the mean across spatial dimensions and channels for each batch item
        mean_noise = torch.mean(noise_map_full, dim=[1, 2, 3], keepdim=True)
        calibrated_noise = (noise_map_full - mean_noise) * intensity

        # Add noise and clip the result
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
