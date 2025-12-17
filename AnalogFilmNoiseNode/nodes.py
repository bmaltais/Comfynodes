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

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C).
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, color noise.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        batch_size, original_height, original_width, num_channels = image.shape
        device = image.device

        # Ensure grain_size is positive
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # ⚡ OPTIMIZATION:
        # The original implementation processed each image in the batch individually,
        # moving data between CPU and GPU within the loop. This is a significant bottleneck.
        #
        # This revised implementation vectorizes the entire process using PyTorch tensor
        # operations, keeping all data on the GPU. This avoids costly CPU-GPU transfers
        # and leverages the GPU's parallel processing capabilities for the whole batch.
        #
        # 1. Generate noise for the entire batch on the correct device.
        # 2. Use torch.nn.functional.interpolate for efficient, hardware-accelerated resizing.
        # 3. Perform all arithmetic (mean calculation, applying noise, clipping) as
        #    vectorized tensor operations.

        # Generate noise tensor (B, C, H, W)
        noise_channels = 1 if monochrome else num_channels
        noise = torch.randn(
            (batch_size, noise_channels, noise_height, noise_width),
            device=device
        )

        # Upscale noise to match image dimensions
        # Permute image to (B, C, H, W) for compatibility with Torch functions
        image_bchw = image.permute(0, 3, 1, 2)

        # Use interpolate for efficient upscaling
        noise_resized = F.interpolate(
            noise,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Replicate noise channels for monochrome
        if monochrome and num_channels > 1:
            noise_resized = noise_resized.repeat(1, num_channels, 1, 1)

        # Calibrate and apply noise
        # Calculate mean per-image in the batch, and apply intensity
        noise_mean = noise_resized.mean(dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_resized - noise_mean) * intensity

        # Add noise and clip
        noisy_image_bchw = torch.clamp(image_bchw + calibrated_noise, 0.0, 1.0)

        # Permute back to (B, H, W, C)
        noisy_image = noisy_image_bchw.permute(0, 2, 3, 1)

        return (noisy_image,)
