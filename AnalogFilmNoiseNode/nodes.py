import torch

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

        # ⚡ Bolt: Vectorized noise generation to avoid slow CPU-GPU roundtrips.
        # The original implementation looped through each image in the batch,
        # converted it to a NumPy array, generated noise on the CPU, and then
        # converted it back to a tensor. This is inefficient.
        #
        # This optimized version performs all operations on the GPU using PyTorch tensors.
        # It generates a single noise tensor for the entire batch, upscales it
        # using `torch.nn.functional.interpolate`, and applies it to the image
        # tensor in a single, vectorized operation. This significantly reduces
        # overhead and leverages GPU parallelism.

        # Determine the shape of the noise tensor.
        if monochrome:
            noise_channels = 1
        else:
            noise_channels = num_channels

        # Generate the noise tensor directly on the GPU.
        # The layout is (N, C, H, W) for compatibility with interpolate.
        noise_shape = (batch_size, noise_channels, noise_height, noise_width)
        noise = torch.randn(noise_shape, dtype=image.dtype, device=image.device)

        # Upscale the noise to match the image dimensions using nearest-neighbor
        # interpolation to create a blocky, film-grain-like appearance.
        upscaled_noise = torch.nn.functional.interpolate(
            noise,
            size=(original_height, original_width),
            mode='nearest'
        )

        # Permute the layout back to (N, H, W, C) to match the input image.
        upscaled_noise = upscaled_noise.permute(0, 2, 3, 1)

        # If the noise is monochrome, repeat it across all channels.
        if monochrome and num_channels > 1:
            upscaled_noise = upscaled_noise.repeat(1, 1, 1, num_channels)

        # Calibrate and apply the noise.
        # We subtract the mean to center the noise distribution around zero.
        calibrated_noise = (upscaled_noise - upscaled_noise.mean()) * intensity
        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)
