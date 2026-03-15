import torch
import numpy as np
import gc
import cv2
import itertools
import comfy.model_management
import comfy.utils
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
import math

# Attempt to import MAX_RESOLUTION from ComfyUI's samplers, with a fallback for safety.
try:
    from comfy.samplers import MAX_RESOLUTION
except ImportError:
    MAX_RESOLUTION = 8192


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
                "intensity": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "grain_size": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "monochrome": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_noise",)
    FUNCTION = "apply_film_noise"
    CATEGORY = "Image/Effects"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, image, intensity, grain_size, monochrome, seed):
        return seed

    def apply_film_noise(
        self,
        image: torch.Tensor,
        intensity: float,
        grain_size: float,
        monochrome: bool,
        seed: int,
    ):
        """
        Adds film grain to the input image using vectorized PyTorch operations.

        Args:
            image (torch.Tensor): The input image tensor in (B, H, W, C) format.
            intensity (float): The strength of the noise effect.
            grain_size (float): The size of the noise grain. Larger values create coarser grain.
            monochrome (bool): If True, applies grayscale noise; otherwise, applies color noise.
            seed (int): Seed for deterministic noise generation.

        Returns:
            (torch.Tensor,): A tuple containing the image tensor with added noise.
        """
        if intensity == 0:
            return (image,)

        device = image.device
        batch_size, original_height, original_width, num_channels = image.shape

        # Ensure grain_size is positive to avoid division by zero
        grain_size = max(0.1, grain_size)

        # Determine noise dimensions based on grain_size.
        noise_height = max(1, int(original_height / grain_size))
        noise_width = max(1, int(original_width / grain_size))

        # Use a PyTorch generator for deterministic noise on the specific device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate noise map (B, C, H, W) for PyTorch operations
        if monochrome:
            noise_map_small = torch.randn(
                (batch_size, 1, noise_height, noise_width),
                generator=generator,
                device=device,
            )
        else:
            noise_map_small = torch.randn(
                (batch_size, num_channels, noise_height, noise_width),
                generator=generator,
                device=device,
            )

        # Upscale noise to match image dimensions
        # Use nearest-neighbor to maintain the blocky grain appearance
        noise_map_full = torch.nn.functional.interpolate(
            noise_map_small,
            size=(original_height, original_width),
            mode="nearest-exact" if grain_size >= 1.0 else "bilinear",
        )

        # If monochrome and image has multiple channels, replicate noise across channels
        if monochrome and num_channels > 1:
            noise_map_full = noise_map_full.repeat(1, num_channels, 1, 1)

        # Reorder to (B, H, W, C) to match input image format
        noise_map_full = noise_map_full.permute(0, 2, 3, 1)

        # Calibrate and apply noise
        # Subtract mean per image to center the noise distribution and scale by intensity
        # Use keepdim=True for proper broadcasting
        noise_mean = noise_map_full.mean(dim=(1, 2, 3), keepdim=True)
        calibrated_noise = (noise_map_full - noise_mean) * intensity

        noisy_image = torch.clamp(image + calibrated_noise, 0.0, 1.0)

        return (noisy_image,)


class ClearGpuMemoryCache:
    """
    A node to clear the GPU's memory cache, freeing up VRAM. It can be used
    to manage memory in complex workflows.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. It accepts any input as a trigger.
        """
        return {
            "required": {
                # This input is a placeholder to ensure the node executes
                # when the workflow reaches this point. It is passed through unmodified.
                "any_type": ("*",)
            }
        }

    # The node passes through the input it receives, without modification.
    RETURN_TYPES = ("*",)
    FUNCTION = "clear_cache"
    OUTPUT_NODE = True
    CATEGORY = "Utilities/Memory"

    def clear_cache(self, any_type):
        """
        Clears the CUDA cache to free up GPU memory and performs garbage collection.

        Args:
            any_type: Any data type, used as a trigger for execution and passed through.

        Returns:
            (any,): A tuple containing the unmodified input data.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return (any_type,)


# HSL conversion functions adapted from https://github.com/limacv/RGB_HSV_HSL
def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.permute(0, 3, 1, 2)
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.0
    hsl_h /= 6.0

    hsl_l = (cmax + cmin) / 2.0
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.0))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (-hsl_l * 2.0 + 2.0))[hsl_l_l0_5]

    hsl = torch.cat([hsl_h, hsl_s, hsl_l], dim=1)
    return hsl.permute(0, 2, 3, 1)


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl = hsl.permute(0, 3, 1, 2)
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2.0 - 1.0) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsl_l - _c / 2.0
    idx = (hsl_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb.permute(0, 2, 3, 1)


class ImageMergeNode:
    """
    A node to merge two images with optional alignment and various blending modes.
    """

    blend_modes = [
        "Normal",
        "Multiply",
        "Screen",
        "Overlay",
        "Soft Light",
        "Color",
        "Darken",
        "Color Burn",
        "Linear Burn",
        "Lighten",
        "Color Dodge",
        "Linear Dodge (Add)",
        "Hard Light",
        "Vivid Light",
        "Linear Light",
        "Pin Light",
        "Hard Mix",
        "Difference",
        "Exclusion",
        "Subtract",
        "Divide",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "original_image": ("IMAGE",),
                "updated_image": ("IMAGE",),
                "blending_mode": (cls.blend_modes,),
                "mixing_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "enable_alignment": ("BOOLEAN", {"default": False}),
                "enable_facial_correction": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "image/layering"

    def _tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts a torch tensor (B, H, W, C) to an OpenCV image (H, W, C, BGR)."""
        np_image = tensor.squeeze(0).cpu().numpy()
        return cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _cv2_to_tensor(self, np_image: np.ndarray) -> torch.Tensor:
        """Converts an OpenCV image (H, W, C, BGR) back to a torch tensor (B, H, W, C)."""
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np_image.astype(np.float32) / 255.0).unsqueeze(0)

    def _align_images(
        self, original_cv2: np.ndarray, updated_cv2: np.ndarray
    ) -> np.ndarray:
        """Aligns the updated image to the original image using feature matching to find the translation."""
        try:
            orb = cv2.ORB_create(nfeatures=1500)
            kp1, des1 = orb.detectAndCompute(original_cv2, None)
            kp2, des2 = orb.detectAndCompute(updated_cv2, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                print(
                    "ImageMergeNode: Not enough descriptors to align. Skipping alignment."
                )
                return updated_cv2

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[: max(20, int(len(matches) * 0.20))]

            if len(good_matches) < 10:
                print(
                    "ImageMergeNode: Not enough good matches to find translation. Skipping alignment."
                )
                return updated_cv2

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 2
            )

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is None:
                print(
                    "ImageMergeNode: Could not compute homography. Skipping alignment."
                )
                return updated_cv2

            h, w = original_cv2.shape[:2]
            aligned_updated_cv2 = cv2.warpPerspective(
                updated_cv2,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            print("ImageMergeNode: Aligned image with perspective transformation.")
            return aligned_updated_cv2

        except Exception as e:
            print(f"ImageMergeNode: Error during alignment: {e}. Skipping alignment.")
            return updated_cv2

    def _get_facial_landmarks(self, image_cv2, face_mesh):
        """Detects facial landmarks in a single image."""
        h, w, _ = image_cv2.shape
        results = face_mesh.process(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        all_landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
                    dtype=np.float32,
                )
                all_landmarks.append(landmarks)
        return all_landmarks

    def _find_and_warp_faces(self, original_cv2, updated_cv2):
        """Finds and warps faces from the updated image to match the original image."""
        try:
            import mediapipe as mp

            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5
            ) as face_mesh:
                original_landmarks_list = self._get_facial_landmarks(
                    original_cv2, face_mesh
                )
                updated_landmarks_list = self._get_facial_landmarks(
                    updated_cv2, face_mesh
                )

                if not original_landmarks_list or not updated_landmarks_list:
                    print(
                        "ImageMergeNode: No faces detected in one or both images. Skipping facial correction."
                    )
                    return updated_cv2

                print(
                    f"ImageMergeNode: Found {len(original_landmarks_list)} face(s) in original and {len(updated_landmarks_list)} in updated."
                )

                final_image = updated_cv2.copy()

                # Use a comprehensive set of landmarks for detailed warping
                key_landmarks_indices = list(
                    itertools.chain(
                        *mp.solutions.face_mesh.FACEMESH_LIPS,
                        *mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                        *mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                        *mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                        *mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
                        *mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
                    )
                )
                if hasattr(mp.solutions.face_mesh, "FACEMESH_NOSE"):
                    key_landmarks_indices += list(
                        itertools.chain(*mp.solutions.face_mesh.FACEMESH_NOSE)
                    )

                key_landmarks_indices = sorted(list(set(key_landmarks_indices)))

                # Match faces based on proximity
                for i, updated_landmarks in enumerate(updated_landmarks_list):
                    updated_center = updated_landmarks.mean(axis=0)
                    distances = [
                        np.linalg.norm(updated_center - orig.mean(axis=0))
                        for orig in original_landmarks_list
                    ]
                    best_match_idx = np.argmin(distances)
                    original_landmarks = original_landmarks_list[best_match_idx]

                    print(
                        f"ImageMergeNode: Warping face {i + 1} in updated to match face {best_match_idx + 1} in original."
                    )

                    # Ensure we have enough landmarks for the detailed set
                    if (
                        original_landmarks.shape[0] < max(key_landmarks_indices) + 1
                        or updated_landmarks.shape[0] < max(key_landmarks_indices) + 1
                    ):
                        print(
                            "ImageMergeNode: Not enough landmarks for detailed warping. Using all available."
                        )
                        source_pts = original_landmarks
                        target_pts = updated_landmarks
                    else:
                        source_pts = np.array(
                            [original_landmarks[j] for j in key_landmarks_indices],
                            dtype=np.float32,
                        )
                        target_pts = np.array(
                            [updated_landmarks[j] for j in key_landmarks_indices],
                            dtype=np.float32,
                        )

                    tps = cv2.createThinPlateSplineShapeTransformer()
                    source_pts_reshaped = source_pts.reshape(1, -1, 2)
                    target_pts_reshaped = target_pts.reshape(1, -1, 2)
                    matches = [cv2.DMatch(i, i, 0) for i in range(len(source_pts))]
                    tps.estimateTransformation(
                        target_pts_reshaped, source_pts_reshaped, matches
                    )

                    warped_updated_cv2 = tps.warpImage(updated_cv2)

                    # Create a mask for the face in the original image to blend
                    hull_indices = cv2.convexHull(
                        original_landmarks, returnPoints=False
                    )
                    hull_points = np.array(
                        [original_landmarks[i[0]] for i in hull_indices], dtype=np.int32
                    )

                    mask = np.zeros(original_cv2.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull_points, 255)

                    # Dilate mask for smoother blending
                    kernel = np.ones((10, 10), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    r = cv2.boundingRect(hull_points)
                    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

                    final_image = self._seamless_clone(
                        warped_updated_cv2, final_image, mask, center
                    )

                return final_image

        except Exception as e:
            print(
                f"ImageMergeNode: Error during facial correction: {e}. Skipping correction."
            )
            return updated_cv2

    def _seamless_clone(self, src, dst, mask, center):
        """Performs seamless cloning to blend the warped face."""
        try:
            return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print(
                f"ImageMergeNode: Error during seamless cloning: {e}. Returning destination image."
            )
            return dst

    def _blend_images_pytorch(
        self, base: torch.Tensor, blend: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """Applies a blending mode to two images using PyTorch."""
        if base.shape[1:3] != blend.shape[1:3]:
            blend = torch.nn.functional.interpolate(
                blend.permute(0, 3, 1, 2),
                size=base.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        base_rgb = base[..., :3]
        blend_rgb = blend[..., :3]

        if mode == "Normal":
            result = blend_rgb
        elif mode == "Multiply":
            result = base_rgb * blend_rgb
        elif mode == "Screen":
            result = 1 - (1 - base_rgb) * (1 - blend_rgb)
        elif mode == "Overlay":
            result = torch.where(
                base_rgb <= 0.5,
                2 * base_rgb * blend_rgb,
                1 - 2 * (1 - base_rgb) * (1 - blend_rgb),
            )
        elif mode == "Soft Light":
            result = torch.where(
                blend_rgb <= 0.5,
                2 * base_rgb * blend_rgb + base_rgb**2 * (1 - 2 * blend_rgb),
                2 * base_rgb * (1 - blend_rgb)
                + torch.sqrt(base_rgb) * (2 * blend_rgb - 1),
            )
        elif mode == "Hard Light":
            result = torch.where(
                blend_rgb <= 0.5,
                2 * base_rgb * blend_rgb,
                1 - 2 * (1 - base_rgb) * (1 - blend_rgb),
            )
        elif mode == "Color":
            base_hsl = rgb2hsl_torch(base_rgb)
            blend_hsl = rgb2hsl_torch(blend_rgb)
            result_hsl = torch.cat((blend_hsl[..., 0:2], base_hsl[..., 2:3]), dim=-1)
            result = hsl2rgb_torch(result_hsl)
        elif mode == "Darken":
            result = torch.min(base_rgb, blend_rgb)
        elif mode == "Color Burn":
            result = 1 - (1 - base_rgb) / (blend_rgb + 1e-6)
        elif mode == "Linear Burn":
            result = base_rgb + blend_rgb - 1
        elif mode == "Lighten":
            result = torch.max(base_rgb, blend_rgb)
        elif mode == "Color Dodge":
            result = base_rgb / (1 - blend_rgb + 1e-6)
        elif mode == "Linear Dodge (Add)":
            result = base_rgb + blend_rgb
        elif mode == "Vivid Light":
            result = torch.where(
                blend_rgb <= 0.5,
                1 - (1 - base_rgb) / (2 * blend_rgb + 1e-6),
                base_rgb / (2 * (1 - blend_rgb) + 1e-6),
            )
        elif mode == "Linear Light":
            result = base_rgb + 2 * blend_rgb - 1
        elif mode == "Pin Light":
            result = torch.where(
                blend_rgb <= 0.5,
                torch.min(base_rgb, 2 * blend_rgb),
                torch.max(base_rgb, 2 * blend_rgb - 1),
            )
        elif mode == "Hard Mix":
            result = torch.floor(base_rgb + blend_rgb)
        elif mode == "Difference":
            result = torch.abs(base_rgb - blend_rgb)
        elif mode == "Exclusion":
            result = base_rgb + blend_rgb - 2 * base_rgb * blend_rgb
        elif mode == "Subtract":
            result = base_rgb - blend_rgb
        elif mode == "Divide":
            result = base_rgb / (blend_rgb + 1e-6)
        else:
            result = blend_rgb

        return torch.clamp(result, 0, 1)

    def merge_images(
        self,
        original_image,
        updated_image,
        blending_mode,
        mixing_strength,
        enable_alignment,
        enable_facial_correction,
    ):
        # The 'updated_image' is the one we want to modify to match the 'original_image'
        # The 'original_image' is the reference

        h, w = original_image.shape[1:3]
        if updated_image.shape[1:3] != (h, w):
            updated_image = torch.nn.functional.interpolate(
                updated_image.permute(0, 3, 1, 2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        base_tensor = updated_image
        blend_tensor = original_image

        if enable_alignment or enable_facial_correction:
            image_to_warp_cv2 = self._tensor_to_cv2(base_tensor)
            reference_image_cv2 = self._tensor_to_cv2(blend_tensor)

            warped_and_aligned_cv2 = image_to_warp_cv2.copy()

            if enable_alignment:
                print("ImageMergeNode: Performing global alignment on updated image.")
                warped_and_aligned_cv2 = self._align_images(
                    reference_image_cv2, warped_and_aligned_cv2
                )

            if enable_facial_correction:
                print("ImageMergeNode: Performing facial correction on updated image.")
                warped_and_aligned_cv2 = self._find_and_warp_faces(
                    reference_image_cv2, warped_and_aligned_cv2
                )

            base_tensor = self._cv2_to_tensor(warped_and_aligned_cv2)

        blended_tensor = self._blend_images_pytorch(
            base_tensor, blend_tensor, blending_mode
        )

        final_tensor = (
            base_tensor * (1.0 - mixing_strength) + blended_tensor * mixing_strength
        )

        return (final_tensor,)


class LatentByMegapixelsAndAspectRatio:
    """
    Generates an empty latent image with dimensions calculated based on a target
    megapixel count and a specific aspect ratio.
    """

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "target_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0625,
                        "max": (MAX_RESOLUTION * MAX_RESOLUTION) / (1024 * 1024),
                        "step": 0.1,
                    },
                ),
                "aspect_ratio_width": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                "aspect_ratio_height": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "target_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "TARGET_WIDTH", "TARGET_HEIGHT")
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(
        self,
        target_megapixels,
        aspect_ratio_width,
        aspect_ratio_height,
        batch_size=1,
        target_multiplier=1.0,
    ):
        """
        Calculates dimensions from megapixels and aspect ratio, then creates an empty latent.

        Args:
            target_megapixels (float): The desired total megapixels (e.g., 1.0 for 1024x1024).
            aspect_ratio_width (int): The width component of the aspect ratio.
            aspect_ratio_height (int): The height component of the aspect ratio.
            batch_size (int): The number of latent images to generate.
            target_multiplier (float): A multiplier to calculate target dimensions for other uses.

        Returns:
            (dict, int, int, int, int): A tuple containing the latent tensor, base width,
            base height, target width, and target height.
        """
        # Calculate total pixels from megapixels
        target_total_pixels = target_megapixels * 1024 * 1024

        # Calculate a scaling factor 's' such that (s * W) * (s * H) = total_pixels
        unit_block_area = aspect_ratio_width * aspect_ratio_height
        scaling_factor = math.sqrt(target_total_pixels / unit_block_area)

        # Calculate initial dimensions
        initial_width = aspect_ratio_width * scaling_factor
        initial_height = aspect_ratio_height * scaling_factor

        # Round to the nearest multiple of 8, ensuring a minimum of 8
        width = max(8, round(initial_width / 8.0) * 8)
        height = max(8, round(initial_height / 8.0) * 8)

        # Enforce MAX_RESOLUTION while attempting to maintain aspect ratio
        current_aspect_ratio = width / height if height != 0 else 1.0
        if width > MAX_RESOLUTION:
            width = MAX_RESOLUTION
            height = max(8, round((width / current_aspect_ratio) / 8.0) * 8)
        if height > MAX_RESOLUTION:
            height = MAX_RESOLUTION
            width = max(8, round((height * current_aspect_ratio) / 8.0) * 8)

        # Final cap to ensure dimensions are within limits after adjustments
        width = min(width, MAX_RESOLUTION)
        height = min(height, MAX_RESOLUTION)

        # A common minimum dimension for stable diffusion is 16
        min_pixel_dim = 16
        width = max(min_pixel_dim, width)
        height = max(min_pixel_dim, height)

        # Calculate target dimensions based on the multiplier
        target_width = max(min_pixel_dim, round((width * target_multiplier) / 8.0) * 8)
        target_height = max(
            min_pixel_dim, round((height * target_multiplier) / 8.0) * 8
        )

        # Cap target dimensions by MAX_RESOLUTION
        target_width = min(target_width, MAX_RESOLUTION)
        target_height = min(target_height, MAX_RESOLUTION)

        # Create the empty latent tensor
        latent_width = width // 8
        latent_height = height // 8
        latent = torch.zeros(
            [batch_size, 4, latent_height, latent_width], device=self.device
        )

        actual_megapixels = (width * height) / (1024 * 1024)
        ui_text = f"{width}x{height} ({actual_megapixels:.2f}MP) -> Target: {target_width}x{target_height}"

        return (
            {"samples": latent, "ui": {"text": ui_text}},
            width,
            height,
            target_width,
            target_height,
        )


class UpscaleImageToTotalPixels:
    """
    Upscales an image to a target total pixel count using an upscaling model.
    If the image is already larger than the target, it's downscaled.
    """

    rescale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def __init__(self):
        self.__imageScaler = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(self):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "total_megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1},
                ),
                "rescale_method": (self.rescale_methods,),
                "skip_model_upscale": ("BOOLEAN", {"default": False}),
                "make_divisible_by": (
                    "INT",
                    {"default": 1, "min": 1, "max": 128, "step": 1},
                ),
            }
        }

    def upscale(
        self,
        upscale_model,
        image,
        total_megapixels,
        rescale_method,
        skip_model_upscale,
        make_divisible_by,
    ):
        """
        Performs the upscaling or downscaling with optional divisibility constraints.

        Args:
            upscale_model: The upscaling model to use.
            image (torch.Tensor): The input image tensor.
            total_megapixels (float): The target total megapixels.
            rescale_method (str): The resampling method for scaling.
            skip_model_upscale (bool): If True, skips model-based upscaling.
            make_divisible_by (int): Ensures final dimensions are divisible by this number.

        Returns:
            (torch.Tensor,): A tuple containing the rescaled image tensor.
        """
        samples = image.movedim(-1, 1)
        original_width = samples.shape[3]
        original_height = samples.shape[2]

        target_pixels = total_megapixels * 1024 * 1024
        current_pixels = original_width * original_height

        # --- Step 1: Initial Upscale (if necessary) ---
        if current_pixels < target_pixels:
            if not skip_model_upscale:
                samples = self.__imageScaler.upscale(upscale_model, image)[0].movedim(
                    -1, 1
                )
            else:
                # Scale up to the target pixel count using standard resampling
                ratio = (target_pixels / current_pixels) ** 0.5
                target_width = round(original_width * ratio)
                target_height = round(original_height * ratio)
                samples = comfy.utils.common_upscale(
                    samples, target_width, target_height, rescale_method, "disabled"
                )

        # --- Step 2: Calculate Final Dimensions with Divisibility ---
        current_width = samples.shape[3]
        current_height = samples.shape[2]

        # Determine the dimensions needed to hit the target pixel count while maintaining aspect ratio
        ratio = (target_pixels / (current_width * current_height)) ** 0.5
        adjustable_width = round(current_width * ratio)
        adjustable_height = round(current_height * ratio)

        final_width = adjustable_width
        final_height = adjustable_height

        m = make_divisible_by
        if m > 1:
            w, h = adjustable_width, adjustable_height

            def ceil_m(val, mult):
                return (val + mult - 1) // mult * mult

            def floor_m(val, mult):
                return (val // mult) * mult

            w_rem = w % m
            h_rem = h % m

            if not (w_rem == 0 and h_rem == 0):
                # Candidate 1: one dimension up, one down
                if w_rem > h_rem or (w_rem == h_rem and w >= h):
                    cand_w = ceil_m(w, m)
                    cand_h = floor_m(h, m)
                else:
                    cand_h = ceil_m(h, m)
                    cand_w = floor_m(w, m)

                # Check if candidate 1 meets the minimum pixel requirement
                if cand_w * cand_h >= target_pixels:
                    final_width = cand_w
                    final_height = cand_h
                else:
                    # Candidate 2: both dimensions up
                    final_width = ceil_m(w, m)
                    final_height = ceil_m(h, m)

        # --- Step 3: Final Resize ---
        if final_width != current_width or final_height != current_height:
            samples = comfy.utils.common_upscale(
                samples, final_width, final_height, rescale_method, "disabled"
            )

        samples = samples.movedim(1, -1)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "AnalogFilmNoiseNode": AnalogFilmNoiseNode,
    "ClearGpuMemoryCache": ClearGpuMemoryCache,
    "ImageMergeNode": ImageMergeNode,
    "LatentByMegapixelsAndAspectRatio": LatentByMegapixelsAndAspectRatio,
    "UpscaleImageToTotalPixels": UpscaleImageToTotalPixels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnalogFilmNoiseNode": "🎞️ Analog Film Noise",
    "ClearGpuMemoryCache": "🧹 Clear GPU Memory Cache",
    "ImageMergeNode": "Image Merge (Align & Blend)",
    "LatentByMegapixelsAndAspectRatio": "Latent by Megapixels & Aspect Ratio",
    "UpscaleImageToTotalPixels": "🚀 Upscale Image to Total Pixels",
}
