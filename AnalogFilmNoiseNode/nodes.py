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
        This approach processes the entire batch on the GPU, avoiding slow CPU-GPU data transfers
        and Python loops for significant performance gains.

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

        # Get image dimensions and device. All operations will remain on this device.
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive.
        grain_size = max(0.1, grain_size)

        # Determine the dimensions of the downscaled noise tensor.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # ⚡ OPTIMIZATION: Generate noise for the entire batch on the GPU at once.
        # This avoids iterating through the batch, a major performance bottleneck.
        noise_channels = 1 if monochrome else num_channels
        noise_shape = (batch_size, noise_channels, noise_height, noise_width)

        # Create noise tensor directly on the target device (e.g., CUDA).
        noise_small = torch.randn(noise_shape, device=device)

        # ⚡ OPTIMIZATION: Use PyTorch's interpolate for fast, vectorized upscaling.
        # This is significantly faster than CPU-based numpy/scipy resizing and keeps data on the GPU.
        # The (B, C, H, W) format is required for interpolate.
        noise_resized = torch.nn.functional.interpolate(
            noise_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channel count.
        if monochrome:
            noise_full = noise_resized.repeat(1, num_channels, 1, 1)
        else:
            noise_full = noise_resized

        # Permute noise from (B, C, H, W) to (B, H, W, C) to match the input image format.
        noise_full = noise_full.permute(0, 2, 3, 1)

        # ⚡ OPTIMIZATION: Apply noise to the entire batch in a single, vectorized operation.
        # torch.randn is already centered at 0, so no mean subtraction is needed.
        calibrated_noise = noise_full * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
