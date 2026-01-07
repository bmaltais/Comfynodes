import torch
import torch.nn.functional as F

class AnalogFilmNoiseNode:
    """
    Applies analog film-style noise to an image. This effect simulates the grain
    found in traditional photographic film.
    """
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
        This approach avoids CPU-GPU data transfers and processes the entire batch at once for better performance.

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

        # Performance Optimization: All operations are performed on the GPU using PyTorch tensors
        # to avoid costly CPU-GPU data transfers and to leverage parallel processing.
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise map on the same device as the input image.
        # The noise is generated in (N, C, H, W) format for PyTorch operations.
        noise_channels = 1 if monochrome else num_channels
        noise_shape = (batch_size, noise_channels, noise_height, noise_width)

        # Performance Optimization: Using torch.randn on the correct device avoids CPU-based generation.
        noise_map_small = torch.randn(noise_shape, device=device)

        # Performance Optimization: Upscale noise using torch.nn.functional.interpolate,
        # which is a highly optimized GPU operation. 'nearest' is used for a sharp, blocky grain.
        noise_map_resized = F.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channels.
        if monochrome and num_channels > 1:
            noise_map_resized = noise_map_resized.repeat(1, num_channels, 1, 1)

        # Performance Optimization: Permute image to (N, C, H, W) for vectorized operations.
        # This is a memory-only operation and is very fast.
        image_nchw = image.permute(0, 3, 1, 2)

        # Calibrate and apply noise. The mean subtraction centers the noise distribution.
        # All of these are fast, element-wise tensor operations on the GPU.
        calibrated_noise = (noise_map_resized - noise_map_resized.mean()) * intensity
        noisy_image_nchw = image_nchw + calibrated_noise

        # Clip the result to the valid [0.0, 1.0] range.
        noisy_image_nchw = torch.clamp(noisy_image_nchw, 0.0, 1.0)

        # Permute the image back to the original (N, H, W, C) format.
        noisy_image = noisy_image_nchw.permute(0, 2, 3, 1)

        return (noisy_image,)
