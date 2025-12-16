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
        Adds film grain to the input image using a vectorized PyTorch approach for performance.
        This avoids costly CPU-GPU data transfers and leverages GPU parallelism.

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

        # ⚡ Bolt Optimization: All operations are performed on the GPU using PyTorch tensors
        # to avoid slow CPU roundtrips. The entire batch is processed at once (vectorized).
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the GPU
        # The shape is prepared for Conv2D-style operations (B, C, H, W)
        if monochrome:
            # For monochrome, we generate single-channel noise and later repeat it for all channels
            noise_shape = (batch_size, 1, noise_height, noise_width)
        else:
            # For color, we generate noise for each channel independently
            noise_shape = (batch_size, num_channels, noise_height, noise_width)

        noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale noise to match image dimensions using nearest-neighbor interpolation.
        # This is a GPU-accelerated equivalent of the previous np.kron/resizing logic.
        # We use permute to match the (B, C, H, W) format expected by interpolate.
        noise_map_resized = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, repeat the single noise channel across all image channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_resized.repeat(1, num_channels, 1, 1)
        else:
            noise_map_full = noise_map_resized

        # Permute back to the original (B, H, W, C) format to match the input image
        noise_map_permuted = noise_map_full.permute(0, 2, 3, 1)

        # Calibrate and apply noise across the entire batch at once
        # Center the noise distribution and scale by intensity
        calibrated_noise = (noise_map_permuted - torch.mean(noise_map_permuted)) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
