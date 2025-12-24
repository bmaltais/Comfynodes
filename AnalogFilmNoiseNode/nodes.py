import torch
import numpy as np
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
        This approach avoids per-image loops and CPU-GPU data transfers, significantly
        improving performance by leveraging GPU parallel processing.

        Args:
            image (torch.Tensor): The input image tensor (N, H, W, C).
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain. Larger values create coarser grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # --- Performance Optimization ---
        # The original implementation looped through each image in the batch, transferred it
        # to the CPU, applied noise with NumPy, and then moved it back to the GPU.
        # This is inefficient due to:
        #   1. Iteration: For-loops in Python are slow.
        #   2. Data Transfer: Moving data between CPU and GPU is a major performance bottleneck.
        #
        # This optimized version performs all operations on the entire batch at once using
        # PyTorch tensor operations, keeping all data on the GPU. This is known as
        # vectorization and is significantly faster.

        # Get image dimensions and device
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Permute image from (N, H, W, C) to (N, C, H, W) for PyTorch operations
        image_chw = image.permute(0, 3, 1, 2)

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the GPU
        if monochrome:
            # Create a single noise channel and expand it later.
            noise_shape = (batch_size, 1, noise_height, noise_width)
        else:
            # Create a noise map with the same number of channels as the image.
            noise_shape = (batch_size, num_channels, noise_height, noise_width)

        # torch.randn is faster than torch.normal on some devices
        noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale noise to match image dimensions using nearest-neighbor interpolation
        # This is the PyTorch equivalent of the blocky grain look from the original's np.kron.
        noise_map_full = F.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, num_channels, 1, 1)

        # Calibrate and apply noise
        # Center the noise distribution around 0 and scale by intensity
        # We calculate the mean across spatial dimensions and channels for each image in the batch
        # and subtract it to prevent a shift in brightness.
        noise_map_mean = noise_map_full.mean(dim=[1, 2, 3], keepdim=True)
        calibrated_noise = (noise_map_full - noise_map_mean) * intensity

        # Add noise to the image and clip the result to the valid [0.0, 1.0] range
        noisy_image_chw = torch.clamp(image_chw + calibrated_noise, 0.0, 1.0)

        # Permute the image back to the original (N, H, W, C) format
        noisy_image = noisy_image_chw.permute(0, 2, 3, 1)

        return (noisy_image,)