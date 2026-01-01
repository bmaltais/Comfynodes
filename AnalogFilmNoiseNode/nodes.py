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
        device = image.device

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor on the same device as the input image
        if monochrome:
            # For monochrome noise, create a single channel and repeat it for all channels
            noise_shape = (batch_size, 1, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device).repeat(1, num_channels, 1, 1)
        else:
            # For color noise, create a noise map with the same number of channels as the image
            noise_shape = (batch_size, num_channels, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale the noise map to the original image dimensions using nearest-neighbor interpolation
        # The tensor needs to be in (B, C, H, W) format for interpolate
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Permute the noise map to match the image tensor's layout (B, H, W, C)
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # ⚡ OPTIMIZATION: The following steps are fully vectorized and run on the GPU.
        # This avoids the performance bottleneck of transferring data between CPU and GPU
        # for each image in the batch, which was present in the previous NumPy-based implementation.

        # Calibrate and apply noise
        # Center the noise distribution around 0 and scale by intensity
        calibrated_noise = (noise_map_full - noise_map_full.mean()) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
