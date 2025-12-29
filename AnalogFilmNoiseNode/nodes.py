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
        This avoids costly CPU-GPU data transfers and leverages the GPU for all operations.

        Args:
            image (torch.Tensor): The input image tensor in BHWC format.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        # ⚡ Bolt Optimization: All operations are performed on the GPU using PyTorch,
        # eliminating the need to transfer data to the CPU (e.g., .cpu().numpy()).
        # This is significantly faster than the previous NumPy-based approach.
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Permute image from BHWC to BCHW format for PyTorch's NCHW convention
        image_bchw = image.permute(0, 3, 1, 2)

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # ⚡ Bolt Optimization: Noise generation is now a single, batched GPU operation
        # using torch.randn, which is much faster than looping and using np.random.
        if monochrome:
            # For monochrome, generate single-channel noise and expand it to all channels
            noise_map_small = torch.randn(batch_size, 1, noise_height, noise_width, device=device)
            if num_channels > 1:
                 noise_map_small = noise_map_small.expand(batch_size, num_channels, noise_height, noise_width)
        else:
            # For color, generate independent noise for each channel
            noise_map_small = torch.randn(batch_size, num_channels, noise_height, noise_width, device=device)

        # ⚡ Bolt Optimization: Upscale noise on the GPU using torch.nn.functional.interpolate,
        # which is a highly optimized operation for this purpose, replacing the slower np.kron.
        # Using 'nearest-exact' for precise nearest-neighbor scaling.
        noise_map_resized = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest-exact' # Use nearest-exact for blocky grain
        )

        # ⚡ Bolt Optimization: All calibration and blending is done in a single,
        # vectorized GPU operation, which is much faster than the previous per-image approach.
        # The mean is calculated across spatial dimensions for proper normalization.
        calibrated_noise = (noise_map_resized - torch.mean(noise_map_resized, dim=[2, 3], keepdim=True)) * intensity
        noisy_image_bchw = torch.clamp(image_bchw + calibrated_noise, 0.0, 1.0)

        # Permute back to BHWC format for ComfyUI
        noisy_image_bhwc = noisy_image_bchw.permute(0, 2, 3, 1)

        return (noisy_image_bhwc,)
