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

        # Vectorized implementation for performance.
        # Moves all operations to the GPU and processes the entire batch at once,
        # avoiding slow CPU-GPU data transfers and per-image loops.
        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive to avoid division by zero or invalid dimensions.
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Generate the noise tensor directly on the GPU.
        if monochrome:
            # Create single-channel noise and expand it to match the image's channel count.
            noise_shape = (batch_size, 1, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device).expand(-1, num_channels, -1, -1)
        else:
            # Generate multi-channel noise for color grain.
            noise_shape = (batch_size, num_channels, noise_height, noise_width)
            noise_map_small = torch.randn(noise_shape, device=device)

        # Upscale the noise map to the original image dimensions using nearest-neighbor interpolation.
        # This is a GPU-accelerated equivalent of np.kron or manual upsampling.
        noise_map_full = F.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode='nearest'
        )

        # The input image is (B, H, W, C), but torch operations are often (B, C, H, W).
        image_bchw = image.permute(0, 3, 1, 2)

        # Calibrate and apply the noise.
        # The noise is centered around 0 and scaled by the intensity.
        calibrated_noise = (noise_map_full - noise_map_full.mean()) * intensity
        noisy_image_bchw = torch.clamp(image_bchw + calibrated_noise, 0.0, 1.0)

        # Permute back to the original (B, H, W, C) format.
        noisy_image_bhwc = noisy_image_bchw.permute(0, 2, 3, 1)

        return (noisy_image_bhwc,)
