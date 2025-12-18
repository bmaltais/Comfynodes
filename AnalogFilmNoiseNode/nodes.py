import torch
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
        Adds film grain to the input image using a vectorized PyTorch implementation.
        This approach avoids CPU-GPU data transfers and processes the entire batch at once for performance.

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

        # Get image dimensions
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device
        dtype = image.dtype

        # ⚡ OPTIMIZATION: Vectorized Implementation
        # The original implementation iterated through the batch, moving data between CPU and GPU,
        # which is a significant performance bottleneck. This version uses pure PyTorch tensor
        # operations to process the entire batch on the GPU at once.

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine the dimensions for the downscaled noise map
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the target device
        # ⚡ OPTIMIZATION: Generate noise for the entire batch and all channels at once
        noise_channels = 1 if monochrome else num_channels
        noise_map_small = torch.randn(
            (batch_size, noise_height, noise_width, noise_channels),
            dtype=dtype,
            device=device
        )

        # ⚡ OPTIMIZATION: Use torch.nn.functional.interpolate for efficient, GPU-accelerated upscaling
        # Permute to (B, C, H, W) for `interpolate`
        noise_map_small = noise_map_small.permute(0, 3, 1, 2)
        noise_map_resized = F.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )
        # Permute back to (B, H, W, C)
        noise_map_full = noise_map_resized.permute(0, 2, 3, 1)

        # If monochrome, replicate the single noise channel across all image channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, 1, 1, num_channels)

        # Calibrate and apply noise across the batch
        # ⚡ OPTIMIZATION: All calculations are now batch-level tensor operations
        # Subtract mean to center the noise distribution
        mean_noise = torch.mean(noise_map_full, dim=[1, 2, 3], keepdim=True)
        calibrated_noise = (noise_map_full - mean_noise) * intensity

        # Add noise to the original image and clip the result
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
