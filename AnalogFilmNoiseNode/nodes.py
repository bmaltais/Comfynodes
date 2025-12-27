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
        Adds film grain to the input image using vectorized PyTorch operations for performance.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C) on the GPU.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # ⚡ Bolt Optimization: Vectorized the entire operation using PyTorch to avoid
        # slow, per-image processing on the CPU. This eliminates the CPU/GPU data
        # transfer bottleneck and leverages GPU parallelization.

        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Determine noise dimensions based on grain_size
        grain_size = max(0.1, grain_size)
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor for the entire batch on the GPU
        # The shape is (B, C, H, W) for compatibility with torch.nn.functional.interpolate
        if monochrome:
            noise_channels = 1
        else:
            noise_channels = num_channels

        noise_shape = (batch_size, noise_channels, noise_height, noise_width)
        noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale noise to match image dimensions using nearest-neighbor interpolation
        # This is significantly faster than the previous np.kron method
        noise_map_resized = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Replicate noise channels for monochrome mode
        if monochrome and num_channels > 1:
            noise_map_resized = noise_map_resized.repeat(1, num_channels, 1, 1)

        # Permute noise from (B, C, H, W) to (B, H, W, C) to match image tensor layout
        noise_map_full = noise_map_resized.permute(0, 2, 3, 1)

        # Calibrate and apply noise directly on the GPU
        # Broadcasting handles the per-image mean calculation efficiently
        mean = torch.mean(noise_map_full, dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - mean) * intensity

        # Add noise and clip the result between 0.0 and 1.0
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
