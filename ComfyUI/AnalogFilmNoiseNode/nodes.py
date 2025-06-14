import torch
import numpy as np

class AnalogFilmNoiseNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}), # Increased max for more effect
                "monochrome": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_noise",)
    FUNCTION = "apply_film_noise"
    CATEGORY = "Image/Effects"
    OUTPUT_NODE = False

    def apply_film_noise(self, image: torch.Tensor, intensity: float, grain_size: float, monochrome: bool):
        if intensity == 0:
            return (image,)

        batch_size, original_height, original_width, num_channels = image.shape

        # Determine noise dimensions based on grain_size
        # A grain_size of 1 means noise is generated at original resolution
        # A grain_size > 1 means noise is generated at a lower resolution and then upscaled
        # A grain_size < 1 means noise is generated at a higher resolution (less common for grain simulation, clamp to avoid issues)
        grain_size = max(0.1, grain_size) # Ensure grain_size is positive

        noise_height = int(original_height / grain_size)
        noise_width = int(original_width / grain_size)

        if noise_height == 0: noise_height = 1
        if noise_width == 0: noise_width = 1

        images_with_noise = []

        for i in range(batch_size):
            img_pil = image[i].cpu().numpy() # Convert to NumPy array (H, W, C)

            if monochrome:
                # Generate single channel noise and replicate across channels
                noise_map_small = np.random.normal(loc=0.0, scale=1.0, size=(noise_height, noise_width, 1))
                if grain_size != 1.0:
                    # Upscale using nearest neighbor to maintain blocky grain appearance
                    noise_map_resized = np.kron(noise_map_small, np.ones((int(grain_size), int(grain_size), 1))) if grain_size >=1 else noise_map_small
                    # Trim if upscaled size is larger than original due to rounding
                    noise_map_full = noise_map_resized[:original_height, :original_width, :]
                else:
                    noise_map_full = noise_map_small

                # Ensure noise_map_full has 3 channels for broadcasting if img_pil has 3 channels
                if num_channels > 1 and noise_map_full.shape[2] == 1:
                     noise_map_full = np.repeat(noise_map_full, num_channels, axis=2)

            else: # Color noise
                # Generate noise for each channel independently
                noise_map_small = np.random.normal(loc=0.0, scale=1.0, size=(noise_height, noise_width, num_channels))
                if grain_size != 1.0:
                    noise_map_resized = np.kron(noise_map_small, np.ones((int(grain_size), int(grain_size), 1))) if grain_size >= 1 else noise_map_small
                    noise_map_full = noise_map_resized[:original_height, :original_width, :]
                else:
                    noise_map_full = noise_map_small

            # Adjust noise shape if it became too large after np.kron
            if noise_map_full.shape[0] > original_height:
                noise_map_full = noise_map_full[:original_height, :, :]
            if noise_map_full.shape[1] > original_width:
                noise_map_full = noise_map_full[:, :original_width, :]

            # Ensure noise_map_full has the same number of channels as the input image
            if noise_map_full.shape[2] != num_channels:
                 noise_map_full = np.repeat(noise_map_full[:,:,0:1], num_channels, axis=2)


            # Apply noise: image is [0,1], noise is N(0,1)
            # Noisy image = image + noise * intensity
            # We scale noise to be roughly [-0.5, 0.5] * intensity to avoid excessive clipping
            # A common approach is to ensure noise doesn't change average brightness much
            calibrated_noise = (noise_map_full - np.mean(noise_map_full)) * intensity

            noisy_img_np = img_pil + calibrated_noise
            noisy_img_np = np.clip(noisy_img_np, 0.0, 1.0)

            images_with_noise.append(torch.from_numpy(noisy_img_np).float().cpu())

        return (torch.stack(images_with_noise),)
