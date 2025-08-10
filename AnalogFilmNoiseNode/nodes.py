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

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        # A larger grain_size results in lower-resolution noise, which is then upscaled.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        images_with_noise = []
        for i in range(batch_size):
            img_np = image[i].cpu().numpy()

            # Generate noise map
            if monochrome:
                # Generate single-channel noise and replicate it across all channels for grayscale grain
                noise_map_small = np.random.normal(loc=0.0, scale=1.0, size=(noise_height, noise_width, 1))
            else:
                # Generate independent noise for each channel for color grain
                noise_map_small = np.random.normal(loc=0.0, scale=1.0, size=(noise_height, noise_width, num_channels))

            # Upscale noise to match image dimensions using nearest-neighbor to maintain the blocky grain appearance
            if grain_size != 1.0 and grain_size >= 1:
                # Use np.kron for a fast nearest-neighbor style upscale
                noise_map_resized = np.kron(noise_map_small, np.ones((int(grain_size), int(grain_size), 1)))
            else:
                noise_map_resized = noise_map_small

            # Trim the upscaled noise map to the exact original image dimensions
            noise_map_full = noise_map_resized[:original_height, :original_width, :]

            # Ensure the noise map has the correct number of channels
            if num_channels > 1 and noise_map_full.shape[2] == 1 and monochrome:
                noise_map_full = np.repeat(noise_map_full, num_channels, axis=2)
            elif noise_map_full.shape[2] != num_channels:
                # Fallback to ensure channel count matches
                noise_map_full = np.repeat(noise_map_full[:, :, 0:1], num_channels, axis=2)

            # Calibrate and apply noise
            # Center the noise distribution and scale by intensity
            calibrated_noise = (noise_map_full - np.mean(noise_map_full)) * intensity
            noisy_img_np = np.clip(img_np + calibrated_noise, 0.0, 1.0)

            images_with_noise.append(torch.from_numpy(noisy_img_np).float().cpu())

        return (torch.stack(images_with_noise),)
