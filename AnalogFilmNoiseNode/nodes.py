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
        Adds film grain to the input image using a fully-vectorized PyTorch implementation.
        This avoids costly CPU-GPU data transfers and leverages GPU parallelism by processing
        the entire batch of images at once.

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

        # Performance: Keep all operations on the GPU to avoid CPU-GPU synchronization stalls.
        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise map for the entire batch on the GPU
        if monochrome:
            # Generate single-channel noise and expand it to all channels
            # Shape: (B, 1, H, W) -> (B, C, H, W)
            noise_map_small = torch.randn(batch_size, 1, noise_height, noise_width, device=device)
            noise_map_small = noise_map_small.repeat(1, num_channels, 1, 1)
        else:
            # Generate independent noise for each channel
            # Shape: (B, C, H, W)
            noise_map_small = torch.randn(batch_size, num_channels, noise_height, noise_width, device=device)

        # Upscale noise to match image dimensions using GPU-accelerated interpolation
        # Using 'nearest' mode to replicate the blocky grain appearance of np.kron
        # Shape: (B, C, H, W) -> (B, C, H_full, W_full)
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # The image tensor is (B, H, W, C), but interpolate output is (B, C, H, W).
        # Permute the noise map to match the image tensor's layout.
        # Shape: (B, C, H, W) -> (B, H, W, C)
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # Calibrate and apply noise across the entire batch
        # Center the noise distribution and scale by intensity
        calibrated_noise = (noise_map_full - noise_map_full.mean()) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
