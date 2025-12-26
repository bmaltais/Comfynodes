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
        Adds film grain to the input image using a fully vectorized GPU-accelerated method.
        This optimization avoids slow CPU-GPU data transfers and leverages PyTorch's parallel
        processing capabilities for a significant performance increase over the previous
        loop-based NumPy implementation.

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

        # Move tensor to the device of the input image tensor
        device = image.device

        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # --- Vectorized GPU implementation ---

        # 1. Generate noise on the GPU
        if monochrome:
            # Generate single-channel noise and expand it to all channels
            noise_channels = 1
        else:
            # Generate independent noise for each channel
            noise_channels = num_channels

        # Create noise tensor for the entire batch
        noise_map_small = torch.randn(
            (batch_size, noise_height, noise_width, noise_channels),
            dtype=image.dtype,
            device=device
        )

        # 2. Upscale noise on the GPU
        # Permute to (N, C, H, W) for interpolate
        noise_map_small_permuted = noise_map_small.permute(0, 3, 1, 2)

        # Use nearest-neighbor interpolation to scale up the noise
        noise_map_full_permuted = F.interpolate(
            noise_map_small_permuted,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Permute back to (N, H, W, C)
        noise_map_full = noise_map_full_permuted.permute(0, 2, 3, 1)

        # 3. Ensure correct channel count for monochrome noise
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.expand(-1, -1, -1, num_channels)

        # 4. Calibrate and apply noise
        # Subtract mean for zero-centering, then scale by intensity
        # We calculate mean per-image in the batch across spatial and channel dims
        noise_mean = torch.mean(noise_map_full, dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - noise_mean) * intensity

        # Add noise to the original image
        noisy_image = image + calibrated_noise

        # 5. Clip the result
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

        return (noisy_image,)
