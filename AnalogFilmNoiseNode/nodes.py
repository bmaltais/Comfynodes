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
        This avoids slow CPU operations and GPU-CPU data transfers, significantly
        improving performance, especially for large batches.

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

        # Get image dimensions and device
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # --- Vectorized Noise Generation (on GPU) ---

        # Generate noise map in (B, C, H, W) format for compatibility with PyTorch's optimized functions.
        if monochrome:
            # Generate single-channel noise and expand it across channels later
            noise_channels = 1
            noise_map_small = torch.randn(
                batch_size, noise_channels, noise_height, noise_width, device=device
            )
        else:
            # Generate independent noise for each channel
            noise_channels = num_channels
            noise_map_small = torch.randn(
                batch_size, noise_channels, noise_height, noise_width, device=device
            )

        # --- Vectorized Upscaling (on GPU) ---

        # Upscale noise to match image dimensions using nearest-neighbor interpolation.
        # This is faster than np.kron and operates entirely on the GPU.
        noise_map_full = F.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # If monochrome, expand the single noise channel to match the image's channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.expand(-1, num_channels, -1, -1)

        # Permute noise from (B, C, H, W) to (B, H, W, C) to match the input image format
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # --- Vectorized Noise Application (on GPU) ---

        # Calibrate and apply noise
        # Center the noise distribution (mean=0) and scale by intensity
        calibrated_noise = (noise_map_full - noise_map_full.mean(dim=[1, 2, 3], keepdim=True)) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
