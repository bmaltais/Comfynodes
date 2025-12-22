import torch
import numpy as np
import cv2
import mediapipe as mp
import itertools


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
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]

    hsl = torch.cat([hsl_h, hsl_s, hsl_l], dim=1)
    return hsl.permute(0, 2, 3, 1)

def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl = hsl.permute(0, 3, 1, 2)
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
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
        "Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Color", "Darken",
        "Color Burn", "Linear Burn", "Lighten", "Color Dodge", "Linear Dodge (Add)",
        "Hard Light", "Vivid Light", "Linear Light", "Pin Light", "Hard Mix",
        "Difference", "Exclusion", "Subtract", "Divide"
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

                final_image = updated_cv2.copy()

                # Use a comprehensive set of landmarks for detailed warping
                key_landmarks_indices = list(itertools.chain(
                    *mp.solutions.face_mesh.FACEMESH_LIPS,
                    *mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    *mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                    *mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    *mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
                    *mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
                ))
                if hasattr(mp.solutions.face_mesh, 'FACEMESH_NOSE'):
                    key_landmarks_indices += list(itertools.chain(*mp.solutions.face_mesh.FACEMESH_NOSE))

                key_landmarks_indices = sorted(list(set(key_landmarks_indices)))

                # Match faces based on proximity
                for i, updated_landmarks in enumerate(updated_landmarks_list):
                    updated_center = updated_landmarks.mean(axis=0)
                    distances = [np.linalg.norm(updated_center - orig.mean(axis=0)) for orig in original_landmarks_list]
                    best_match_idx = np.argmin(distances)
                    original_landmarks = original_landmarks_list[best_match_idx]

                    print(f"ImageMergeNode: Warping face {i+1} in updated to match face {best_match_idx+1} in original.")

                    # Ensure we have enough landmarks for the detailed set
                    if original_landmarks.shape[0] < max(key_landmarks_indices) + 1 or \
                       updated_landmarks.shape[0] < max(key_landmarks_indices) + 1:
                        print("ImageMergeNode: Not enough landmarks for detailed warping. Using all available.")
                        source_pts = original_landmarks
                        target_pts = updated_landmarks
                    else:
                        source_pts = np.array([original_landmarks[j] for j in key_landmarks_indices], dtype=np.float32)
                        target_pts = np.array([updated_landmarks[j] for j in key_landmarks_indices], dtype=np.float32)

                    tps = cv2.createThinPlateSplineShapeTransformer()
                    source_pts_reshaped = source_pts.reshape(1, -1, 2)
                    target_pts_reshaped = target_pts.reshape(1, -1, 2)
                    matches = [cv2.DMatch(i, i, 0) for i in range(len(source_pts))]
                    tps.estimateTransformation(target_pts_reshaped, source_pts_reshaped, matches)

                    warped_updated_cv2 = tps.warpImage(updated_cv2)

                    # Create a mask for the face in the original image to blend
                    hull_indices = cv2.convexHull(original_landmarks, returnPoints=False)
                    hull_points = np.array([original_landmarks[i[0]] for i in hull_indices], dtype=np.int32)

                    mask = np.zeros(original_cv2.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull_points, 255)

                    # Dilate mask for smoother blending
                    kernel = np.ones((10, 10), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    r = cv2.boundingRect(hull_points)
                    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

                    final_image = self._seamless_clone(warped_updated_cv2, final_image, mask, center)

                return final_image

        except Exception as e:
            print(f"ImageMergeNode: Error during facial correction: {e}. Skipping correction.")
            return updated_cv2

    def _seamless_clone(self, src, dst, mask, center):
        """Performs seamless cloning to blend the warped face."""
        try:
            return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"ImageMergeNode: Error during seamless cloning: {e}. Returning destination image.")
            return dst

    def _blend_images_pytorch(self, base: torch.Tensor, blend: torch.Tensor, mode: str) -> torch.Tensor:
        """Applies a blending mode to two images using PyTorch."""
        if base.shape[1:3] != blend.shape[1:3]:
            blend = torch.nn.functional.interpolate(blend.permute(0, 3, 1, 2), size=base.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        base_rgb = base[..., :3]
        blend_rgb = blend[..., :3]

        if mode == 'Normal':
            result = blend_rgb
        elif mode == 'Multiply':
            result = base_rgb * blend_rgb
        elif mode == 'Screen':
            result = 1 - (1 - base_rgb) * (1 - blend_rgb)
        elif mode == 'Overlay':
            result = torch.where(base_rgb <= 0.5, 2 * base_rgb * blend_rgb, 1 - 2 * (1 - base_rgb) * (1 - blend_rgb))
        elif mode == 'Soft Light':
            result = torch.where(blend_rgb <= 0.5, 2 * base_rgb * blend_rgb + base_rgb**2 * (1 - 2 * blend_rgb), 2 * base_rgb * (1 - blend_rgb) + torch.sqrt(base_rgb) * (2 * blend_rgb - 1))
        elif mode == 'Hard Light':
            result = torch.where(blend_rgb <= 0.5, 2 * base_rgb * blend_rgb, 1 - 2 * (1 - base_rgb) * (1 - blend_rgb))
        elif mode == 'Color':
            base_hsl = rgb2hsl_torch(base_rgb)
            blend_hsl = rgb2hsl_torch(blend_rgb)
            result_hsl = torch.cat((blend_hsl[..., 0:2], base_hsl[..., 2:3]), dim=-1)
            result = hsl2rgb_torch(result_hsl)
        elif mode == 'Darken':
            result = torch.min(base_rgb, blend_rgb)
        elif mode == 'Color Burn':
            result = 1 - (1 - base_rgb) / (blend_rgb + 1e-6)
        elif mode == 'Linear Burn':
            result = base_rgb + blend_rgb - 1
        elif mode == 'Lighten':
            result = torch.max(base_rgb, blend_rgb)
        elif mode == 'Color Dodge':
            result = base_rgb / (1 - blend_rgb + 1e-6)
        elif mode == 'Linear Dodge (Add)':
            result = base_rgb + blend_rgb
        elif mode == 'Vivid Light':
            result = torch.where(blend_rgb <= 0.5, 1 - (1 - base_rgb) / (2 * blend_rgb + 1e-6), base_rgb / (2 * (1 - blend_rgb) + 1e-6))
        elif mode == 'Linear Light':
            result = base_rgb + 2 * blend_rgb - 1
        elif mode == 'Pin Light':
            result = torch.where(blend_rgb <= 0.5, torch.min(base_rgb, 2 * blend_rgb), torch.max(base_rgb, 2 * blend_rgb - 1))
        elif mode == 'Hard Mix':
            result = torch.floor(base_rgb + blend_rgb)
        elif mode == 'Difference':
            result = torch.abs(base_rgb - blend_rgb)
        elif mode == 'Exclusion':
            result = base_rgb + blend_rgb - 2 * base_rgb * blend_rgb
        elif mode == 'Subtract':
            result = base_rgb - blend_rgb
        elif mode == 'Divide':
            result = base_rgb / (blend_rgb + 1e-6)
        else:
            result = blend_rgb

        return torch.clamp(result, 0, 1)

    def merge_images(self, original_image, updated_image, blending_mode, mixing_strength, enable_alignment, enable_facial_correction):
        # The 'updated_image' is the one we want to modify to match the 'original_image'
        # The 'original_image' is the reference

        h, w = original_image.shape[1:3]
        if updated_image.shape[1:3] != (h, w):
            updated_image = torch.nn.functional.interpolate(updated_image.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        base_tensor = updated_image
        blend_tensor = original_image

        if enable_alignment or enable_facial_correction:
            image_to_warp_cv2 = self._tensor_to_cv2(base_tensor)
            reference_image_cv2 = self._tensor_to_cv2(blend_tensor)

            warped_and_aligned_cv2 = image_to_warp_cv2.copy()

            if enable_alignment:
                print("ImageMergeNode: Performing global alignment on updated image.")
                warped_and_aligned_cv2 = self._align_images(reference_image_cv2, warped_and_aligned_cv2)

            if enable_facial_correction:
                print("ImageMergeNode: Performing facial correction on updated image.")
                warped_and_aligned_cv2 = self._find_and_warp_faces(reference_image_cv2, warped_and_aligned_cv2)

            base_tensor = self._cv2_to_tensor(warped_and_aligned_cv2)

        blended_tensor = self._blend_images_pytorch(base_tensor, blend_tensor, blending_mode)

        final_tensor = base_tensor * (1.0 - mixing_strength) + blended_tensor * mixing_strength

        return (final_tensor,)
