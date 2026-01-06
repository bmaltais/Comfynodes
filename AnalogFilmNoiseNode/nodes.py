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
        ⚡ Bolt: Vectorized film grain application.
        This function was refactored to eliminate a slow, per-image loop that transferred
        data between the CPU and GPU. The new implementation is fully vectorized,
        performing all operations on the GPU for a significant performance boost.

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

        # Move tensor to the same device as the input image
        device = image.device
        dtype = image.dtype

        batch_size, original_height, original_width, num_channels = image.shape
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate noise tensor directly on the GPU
        noise_channels = 1 if monochrome else num_channels
        noise_shape = (batch_size, noise_height, noise_width, noise_channels)
        noise = torch.randn(noise_shape, dtype=dtype, device=device)

        # Permute for interpolation: (B, C, H, W)
        noise = noise.permute(0, 3, 1, 2)

        # Upscale noise to match image dimensions
        noise_resized = F.interpolate(
            noise,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Permute back: (B, H, W, C)
        noise_resized = noise_resized.permute(0, 2, 3, 1)

        # Expand monochrome noise to all channels if necessary
        if monochrome and num_channels > 1:
            noise_resized = noise_resized.repeat(1, 1, 1, num_channels)

        # Calibrate and apply noise
        # Note: Using per-channel mean for color noise is more accurate
        noise_mean = noise_resized.mean(dim=[1, 2], keepdim=True)
        calibrated_noise = (noise_resized - noise_mean) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
