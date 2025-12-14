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
        Adds film grain to the input image using vectorized PyTorch operations for performance.

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

        # Performance: Keep all operations on the GPU to avoid costly CPU-GPU data transfers.
        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Performance: Use PyTorch tensor operations instead of a loop over the batch.
        image_bchw = image.permute(0, 3, 1, 2)

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the GPU
        noise_channels = 1 if monochrome else num_channels
        # Performance: `torch.randn` is the GPU-accelerated equivalent of `np.random.normal`.
        noise_map_small = torch.randn(batch_size, noise_channels, noise_height, noise_width, device=device)

        # Upscale noise to match image dimensions
        # Performance: `F.interpolate` is a highly optimized PyTorch function for resizing.
        # Using 'nearest' mode is equivalent to `np.kron` for this purpose but works on batches.
        noise_map_full = F.interpolate(noise_map_small, size=(original_height, original_width), mode='nearest')

        # Replicate noise channel for monochrome effect
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, num_channels, 1, 1)

        # Calibrate and apply noise
        # Performance: Vectorized mean calculation and scaling across the entire batch.
        noise_map_full = noise_map_full - noise_map_full.mean()
        image_bchw = image_bchw + noise_map_full * intensity

        # Performance: `torch.clamp` is the GPU-accelerated equivalent of `np.clip`.
        noisy_image = torch.clamp(image_bchw, 0.0, 1.0)

        # Convert back to (B, H, W, C) format
        return (noisy_image.permute(0, 2, 3, 1),)
