import torch
import numpy as np
import cv2

class ImageMergeNode:
    """
    A node to merge two images with optional alignment and various blending modes.
    """

    blend_modes = ["Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Color"]

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
                "mixing_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_alignment": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "image/layering"

    def _tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """Converts a torch tensor (B, H, W, C) to an OpenCV image (H, W, C, BGR)."""
        np_image = tensor.squeeze(0).cpu().numpy()
        # Ensure conversion from RGB to BGR for OpenCV
        return cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _cv2_to_tensor(self, np_image: np.ndarray) -> torch.Tensor:
        """Converts an OpenCV image (H, W, C, BGR) back to a torch tensor (B, H, W, C)."""
        # Ensure conversion from BGR to RGB for PyTorch
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np_image.astype(np.float32) / 255.0).unsqueeze(0)

    def _align_images(self, original_cv2: np.ndarray, updated_cv2: np.ndarray) -> np.ndarray:
        """Aligns the updated image to the original image using feature matching to find the translation."""
        try:
            # Use ORB to detect keypoints and compute descriptors
            orb = cv2.ORB_create(nfeatures=1500)
            kp1, des1 = orb.detectAndCompute(original_cv2, None)
            kp2, des2 = orb.detectAndCompute(updated_cv2, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                print("ImageMergeNode: Not enough descriptors to align. Skipping alignment.")
                return updated_cv2

            # Use Brute-Force Matcher with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Sort matches by distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)

            # Keep a reasonable number of best matches
            good_matches = matches[:max(20, int(len(matches) * 0.20))]

            if len(good_matches) < 10:
                print("ImageMergeNode: Not enough good matches to find translation. Skipping alignment.")
                return updated_cv2

            # Extract coordinates of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Calculate the median translation vector, which is robust to outliers
            translations = src_pts - dst_pts
            dx = np.median(translations[:, 0])
            dy = np.median(translations[:, 1])

            # Create the transformation matrix for translation
            M = np.float32([[1, 0, dx], [0, 1, dy]])

            # Apply the translation to the updated image
            h, w = original_cv2.shape[:2]
            aligned_updated_cv2 = cv2.warpAffine(updated_cv2, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            print(f"ImageMergeNode: Aligned image with translation (dx: {dx:.2f}, dy: {dy:.2f})")
            return aligned_updated_cv2

        except Exception as e:
            print(f"ImageMergeNode: Error during alignment: {e}. Skipping alignment.")
            return updated_cv2

    def _blend_images(self, base_img: np.ndarray, blend_img: np.ndarray, mode: str) -> np.ndarray:
        """Applies a blending mode to two images."""
        if base_img.shape[:2] != blend_img.shape[:2]:
            # Resize blend_img to match base_img dimensions for blending
            blend_img = cv2.resize(blend_img, (base_img.shape[1], base_img.shape[0]))

        # Convert to float32 for calculations, handling both 3 and 4 channel images
        base = base_img[..., :3].astype(np.float32) / 255.0
        blend = blend_img[..., :3].astype(np.float32) / 255.0

        if mode == 'Normal':
            result = blend
        elif mode == 'Multiply':
            result = base * blend
        elif mode == 'Screen':
            result = 1 - (1 - base) * (1 - blend)
        elif mode == 'Overlay':
            result = np.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
        elif mode == 'Soft Light':
            result = np.where(blend < 0.5,
                              (2 * base * blend + base**2 * (1 - 2 * blend)),
                              (np.sqrt(base) * (2 * blend - 1) + 2 * base * (1 - blend)))
        elif mode == 'Color':
            base_hls = cv2.cvtColor(base_img, cv2.COLOR_BGR2HLS)
            blend_hls = cv2.cvtColor(blend_img, cv2.COLOR_BGR2HLS)
            result_hls = np.dstack((blend_hls[:,:,0], base_hls[:,:,1], blend_hls[:,:,2]))
            result_bgr = cv2.cvtColor(result_hls, cv2.COLOR_HLS2BGR)
            result = result_bgr.astype(np.float32) / 255.0
        else:
            result = blend

        # Clip values to [0, 1] and convert back to uint8
        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result_uint8

    def merge_images(self, original_image, updated_image, blending_mode, mixing_strength, enable_alignment):
        # Base image is the 'updated' one (bottom layer)
        base_cv2 = self._tensor_to_cv2(updated_image)
        # Blend image is the 'original' one (top layer, what we overlay)
        blend_cv2 = self._tensor_to_cv2(original_image)

        if enable_alignment:
            # Align the blend image (original) to the base image (updated)
            h, w = base_cv2.shape[:2]
            blend_cv2_resized = cv2.resize(blend_cv2, (w, h))
            blend_cv2 = self._align_images(base_cv2, blend_cv2_resized)

        # Get the result of the blend mode operation
        blended_cv2 = self._blend_images(base_cv2, blend_cv2, blending_mode)

        # Mix the blended result with the base image using mixing_strength
        # final = (1 - strength) * base + strength * blended
        final_cv2 = cv2.addWeighted(blended_cv2, mixing_strength, base_cv2, 1.0 - mixing_strength, 0)

        # Convert back to tensor for ComfyUI
        result_tensor = self._cv2_to_tensor(final_cv2)

        return (result_tensor,)
