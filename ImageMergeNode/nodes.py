import torch
import numpy as np
import cv2
import mediapipe as mp
import itertools

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

    def _align_images(self, original_cv2: np.ndarray, updated_cv2: np.ndarray) -> np.ndarray:
        """Aligns the updated image to the original image using feature matching to find the translation."""
        try:
            orb = cv2.ORB_create(nfeatures=1500)
            kp1, des1 = orb.detectAndCompute(original_cv2, None)
            kp2, des2 = orb.detectAndCompute(updated_cv2, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                print("ImageMergeNode: Not enough descriptors to align. Skipping alignment.")
                return updated_cv2

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(20, int(len(matches) * 0.20))]

            if len(good_matches) < 10:
                print("ImageMergeNode: Not enough good matches to find translation. Skipping alignment.")
                return updated_cv2

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            M, mask = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is None:
                print("ImageMergeNode: Could not compute affine transformation. Skipping alignment.")
                return updated_cv2

            h, w = original_cv2.shape[:2]
            aligned_updated_cv2 = cv2.warpAffine(updated_cv2, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            print(f"ImageMergeNode: Aligned image with affine transformation.")
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
                landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
                all_landmarks.append(landmarks)
        return all_landmarks

    def _find_and_warp_faces(self, original_cv2, updated_cv2):
        """Finds and warps faces from the updated image to match the original image."""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5) as face_mesh:
                original_landmarks_list = self._get_facial_landmarks(original_cv2, face_mesh)
                updated_landmarks_list = self._get_facial_landmarks(updated_cv2, face_mesh)

                if not original_landmarks_list or not updated_landmarks_list:
                    print("ImageMergeNode: No faces detected in one or both images. Skipping facial correction.")
                    return updated_cv2

                print(f"ImageMergeNode: Found {len(original_landmarks_list)} face(s) in original and {len(updated_landmarks_list)} in updated.")

                warped_image = updated_cv2.copy()

                # Match faces based on proximity of bounding box centers
                for i, updated_landmarks in enumerate(updated_landmarks_list):
                    # Find the closest face in the original image
                    updated_center = updated_landmarks.mean(axis=0)
                    distances = [np.linalg.norm(updated_center - orig.mean(axis=0)) for orig in original_landmarks_list]
                    best_match_idx = np.argmin(distances)
                    original_landmarks = original_landmarks_list[best_match_idx]

                    print(f"ImageMergeNode: Warping face {i+1} in updated image to match face {best_match_idx+1} in original.")

                    # Use a subset of landmarks for more stable warping (e.g., facial outline, eyes, nose, mouth)
                    # These indices are standard across MediaPipe's 468 landmarks
                    key_landmarks_indices = [
                        33, 263, 61, 291, 199, # Face outline
                        362, 385, 387, 263, 373, 380, # Left eye
                        133, 158, 160, 33, 144, 153, # Right eye
                        1, 2, 98, 327, # Nose
                        61, 84, 17, 314, 291, 405, 18, 178 # Mouth
                    ]

                    # Ensure all key landmarks are within the detected landmarks
                    if original_landmarks.shape[0] < max(key_landmarks_indices) + 1:
                         print("ImageMergeNode: Not enough landmarks detected for stable warping. Using all available.")
                         source_pts = original_landmarks
                         target_pts = updated_landmarks
                    else:
                        source_pts = np.array([original_landmarks[j] for j in key_landmarks_indices], dtype=np.float32)
                        target_pts = np.array([updated_landmarks[j] for j in key_landmarks_indices], dtype=np.float32)

                    # Use Thin Plate Spline for warping
                    tps = cv2.createThinPlateSplineShapeTransformer()

                    source_pts_list = [[pt[0], pt[1]] for pt in source_pts]
                    target_pts_list = [[pt[0], pt[1]] for pt in target_pts]

                    matches = [cv2.DMatch(i, i, 0) for i in range(len(source_pts_list))]

                    tps.estimateTransformation(np.array([target_pts_list]), np.array([source_pts_list]), matches)
                    warped_image = tps.warpImage(warped_image)

                return warped_image

        except Exception as e:
            print(f"ImageMergeNode: Error during facial correction: {e}. Skipping correction.")
            return updated_cv2

        return updated_cv2

    def _blend_images(self, base_img: np.ndarray, blend_img: np.ndarray, mode: str) -> np.ndarray:
        """Applies a blending mode to two images."""
        if base_img.shape[:2] != blend_img.shape[:2]:
            blend_img = cv2.resize(blend_img, (base_img.shape[1], base_img.shape[0]))

        base = base_img[..., :3].astype(np.float32) / 255.0
        blend = blend_img[..., :3].astype(np.float32) / 255.0

        if mode == 'Normal': result = blend
        elif mode == 'Multiply': result = base * blend
        elif mode == 'Screen': result = 1 - (1 - base) * (1 - blend)
        elif mode == 'Overlay': result = np.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
        elif mode == 'Soft Light': result = np.where(blend < 0.5, (2 * base * blend + base**2 * (1 - 2 * blend)), (np.sqrt(base) * (2 * blend - 1) + 2 * base * (1 - blend)))
        elif mode == 'Color':
            base_hls = cv2.cvtColor(base_img, cv2.COLOR_BGR2HLS)
            blend_hls = cv2.cvtColor(blend_img, cv2.COLOR_BGR2HLS)
            result_hls = np.dstack((blend_hls[:,:,0], base_hls[:,:,1], blend_hls[:,:,2]))
            result_bgr = cv2.cvtColor(result_hls, cv2.COLOR_HLS2BGR)
            result = result_bgr.astype(np.float32) / 255.0
        else: result = blend

        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result_uint8

    def merge_images(self, original_image, updated_image, blending_mode, mixing_strength, enable_alignment, enable_facial_correction):
        # The 'updated_image' is the one we want to modify to match the 'original_image'
        image_to_warp_cv2 = self._tensor_to_cv2(updated_image)
        # The 'original_image' is the reference
        reference_image_cv2 = self._tensor_to_cv2(original_image)

        # Ensure images are the same size before any processing
        h, w = reference_image_cv2.shape[:2]
        image_to_warp_cv2 = cv2.resize(image_to_warp_cv2, (w, h))

        warped_and_aligned_cv2 = image_to_warp_cv2.copy()

        if enable_alignment:
            # The existing alignment aligns the 'original' to the 'updated'.
            # We want to align the 'updated' to the 'original'.
            # So we align `image_to_warp_cv2` to `reference_image_cv2`.
            print("ImageMergeNode: Performing global alignment on updated image.")
            warped_and_aligned_cv2 = self._align_images(reference_image_cv2, warped_and_aligned_cv2)

        if enable_facial_correction:
            # Now, warp the faces of the (potentially aligned) 'updated' image to match the 'original'
            print("ImageMergeNode: Performing facial correction on updated image.")
            warped_and_aligned_cv2 = self._find_and_warp_faces(reference_image_cv2, warped_and_aligned_cv2)

        # The `base_cv2` for blending should be the final, warped version of the updated image.
        base_cv2 = warped_and_aligned_cv2
        # The `blend_cv2` is the original image, which acts as the top layer in blending.
        blend_cv2 = reference_image_cv2

        # Get the result of the blend mode operation
        blended_cv2 = self._blend_images(base_cv2, blend_cv2, blending_mode)

        # Mix the blended result with the base image using mixing_strength
        final_cv2 = cv2.addWeighted(blended_cv2, mixing_strength, base_cv2, 1.0 - mixing_strength, 0)

        result_tensor = self._cv2_to_tensor(final_cv2)

        return (result_tensor,)
