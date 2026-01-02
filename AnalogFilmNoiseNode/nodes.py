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

        # ⚡ Bolt Optimization: Vectorized film noise generation on GPU
        # This entire operation is moved to the GPU using PyTorch tensors to avoid
        # slow CPU-GPU data transfers and leverage parallel processing.
        # The previous implementation iterated through each image in the batch,
        # converting it to a NumPy array on the CPU, which is a major performance bottleneck.

        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device # Get the device of the input tensor (e.g., 'cuda:0')

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the GPU
        if monochrome:
            # Create single-channel noise and expand it to match image channels
            noise_shape = (batch_size, 1, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device).expand(-1, num_channels, -1, -1)
        else:
            # Create multi-channel noise for color grain
            noise_shape = (batch_size, num_channels, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale noise to match image dimensions using efficient interpolation
        # Permute image from (B, H, W, C) to (B, C, H, W) for PyTorch functions
        image_bchw = image.permute(0, 3, 1, 2)

        # Use interpolate for efficient, hardware-accelerated resizing on the GPU
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Calibrate and apply noise
        # Center the noise distribution (mean=0) and scale by intensity
        # The view(...).mean(...) calculates the mean across H, W, and C for each batch item
        calibrated_noise = (noise_map_full - noise_map_full.view(batch_size, -1).mean(dim=1).view(batch_size, 1, 1, 1)) * intensity

        # Add noise to the image and clamp the result to the valid [0.0, 1.0] range
        noisy_image_bchw = torch.clamp(image_bchw + calibrated_noise, 0.0, 1.0)

        # Permute back to the original (B, H, W, C) format
        noisy_image = noisy_image_bchw.permute(0, 2, 3, 1)

        return (noisy_image,)
