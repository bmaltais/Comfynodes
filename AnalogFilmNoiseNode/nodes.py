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
        This approach keeps all operations on the GPU, avoiding costly CPU-GPU data transfers
        and leveraging parallel processing for the entire batch.

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

        # Performance: Keep all operations on the GPU to avoid CPU bottlenecks.
        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive to avoid division by zero or invalid dimensions.
        grain_size = max(0.1, grain_size)

        # Performance: Calculate noise dimensions once for the entire batch.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Performance: Generate noise for the entire batch at once using torch.randn on the GPU.
        if monochrome:
            # Create single-channel noise and expand it to all channels.
            noise_shape = (batch_size, 1, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device).expand(-1, num_channels, -1, -1)
        else:
            # Create multi-channel noise for colored grain.
            noise_shape = (batch_size, num_channels, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device)

        # Performance: Use torch.nn.functional.interpolate for efficient, GPU-accelerated upscaling.
        # This is faster and more flexible than np.kron and handles non-integer scaling factors correctly.
        # The output from interpolate is (B, C, H, W), which is what we need.
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Performance: Permute dimensions to match the input image (B, H, W, C) for broadcasting.
        # This is a metadata-only operation and is extremely fast.
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # Performance: Calibrate noise on the GPU. The mean is calculated across spatial dimensions
        # and channels for each batch item independently, then scaled by intensity.
        # Using keepdim=True ensures that the dimensions are broadcastable for subtraction.
        mean_noise = torch.mean(noise_map_full, dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - mean_noise) * intensity

        # Apply noise and clip the result in a single, vectorized operation.
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
