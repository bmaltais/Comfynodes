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
        This approach avoids iterating over the batch and keeps all computations on the GPU,
        significantly improving performance by eliminating CPU-GPU data transfers.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C).
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # ⚡ Bolt Optimization: Vectorized the entire operation to run on the GPU.
        # This eliminates a major performance bottleneck caused by iterating through the
        # image batch and performing CPU-bound NumPy operations in a loop. By using
        # PyTorch tensors for noise generation, interpolation, and application, we
        # keep all data on the GPU, avoiding costly CPU-GPU synchronization and data transfers.
        # This results in a significant speedup, especially for large batches or high-res images.

        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        grain_size = max(0.1, grain_size)
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Determine the shape for the noise tensor.
        # For monochrome noise, we generate a single channel and later expand it.
        noise_channels = 1 if monochrome else num_channels
        noise_shape = (batch_size, noise_height, noise_width, noise_channels)

        # Generate the noise tensor directly on the target device (GPU).
        noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale the noise map to the original image dimensions using nearest-neighbor interpolation.
        # The tensor is permuted to (B, C, H, W) format required by interpolate.
        noise_map_small_permuted = noise_map_small.permute(0, 3, 1, 2)
        noise_map_full_permuted = torch.nn.functional.interpolate(
            noise_map_small_permuted,
            size=(original_height, original_width),
            mode='nearest'
        )
        # Permute back to the original (B, H, W, C) format.
        noise_map_full = noise_map_full_permuted.permute(0, 2, 3, 1)

        # If monochrome, expand the single noise channel to match the image's channels.
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.expand(-1, -1, -1, num_channels)

        # Calibrate and apply noise: center the distribution and scale by intensity.
        # We calculate the mean across spatial dimensions for each image and channel in the batch.
        mean_noise = noise_map_full.mean(dim=[1, 2], keepdim=True)
        calibrated_noise = (noise_map_full - mean_noise) * intensity

        # Add the noise to the image and clip the result to the valid [0.0, 1.0] range.
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
