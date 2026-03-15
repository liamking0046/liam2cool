from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class MotionConfig:
    flow_scale: float = 0.8
    global_scale: float = 1.0
    blend_strength: float = 0.15
    fps: float | None = None
    max_frames: int | None = None
    look: str = "natural"
    look_strength: float = 0.45
    motion_focus: float = 0.6
    hand_boost: float = 0.35
    temporal_smooth: float = 0.2
    identity_lock: float = 0.45


class MotionControlGenerator:
    def __init__(self, config: MotionConfig):
        self.config = config

    @staticmethod
    def _read_reference_frames(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open reference video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames: list[np.ndarray] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        if len(frames) < 2:
            raise ValueError("Reference video must contain at least 2 frames.")

        return frames, fps

    @staticmethod
    def _resize_to_reference(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        return cv2.resize(source, (w, h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _estimate_global_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7,
        )

        if prev_pts is None or len(prev_pts) < 8:
            return np.eye(3, dtype=np.float32)

        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or st is None:
            return np.eye(3, dtype=np.float32)

        st = st.reshape(-1) == 1
        p0 = prev_pts[st].reshape(-1, 2)
        p1 = curr_pts[st].reshape(-1, 2)

        if len(p0) < 8:
            return np.eye(3, dtype=np.float32)

        H, _ = cv2.findHomography(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return np.eye(3, dtype=np.float32)

        return H.astype(np.float32)

    @staticmethod
    def _dense_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

    @staticmethod
    def _warp_with_flow(image: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0] * scale).astype(np.float32)
        map_y = (grid_y + flow[..., 1] * scale).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _warp_with_homography(image: np.ndarray, H: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        if np.allclose(H, np.eye(3, dtype=np.float32)):
            return image

        blended_H = np.eye(3, dtype=np.float32) * (1.0 - scale) + H * scale
        blended_H[2, 2] = 1.0

        return cv2.warpPerspective(image, blended_H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _filmic_curve(image: np.ndarray) -> np.ndarray:
        x = image.astype(np.float32) / 255.0
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    @staticmethod
    def _upper_body_prior(height: int, width: int) -> np.ndarray:
        y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)
        x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        torso = np.exp(-((yy - 0.35) ** 2) / 0.10)
        hands_left = np.exp(-(((xx - 0.22) ** 2) + ((yy - 0.45) ** 2)) / 0.035)
        hands_right = np.exp(-(((xx - 0.78) ** 2) + ((yy - 0.45) ** 2)) / 0.035)
        prior = torso + 0.8 * (hands_left + hands_right)
        prior = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 5)
        return np.clip(prior / (np.max(prior) + 1e-6), 0.0, 1.0)

    def _motion_focus_mask(self, flow: np.ndarray, height: int, width: int) -> np.ndarray:
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        p95 = np.percentile(magnitude, 95)
        normalized = magnitude / (p95 + 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)

        prior = self._upper_body_prior(height, width)
        focus = float(np.clip(self.config.motion_focus, 0.0, 1.0))
        hand_boost = float(np.clip(self.config.hand_boost, 0.0, 1.0))

        emphasized = normalized * (1.0 + focus * prior + hand_boost * np.sqrt(prior))
        emphasized = np.clip(emphasized, 0.0, 1.0)
        mask = cv2.GaussianBlur(emphasized.astype(np.float32), (0, 0), 3)
        return np.clip(mask, 0.0, 1.0)

    def _apply_identity_lock(self, frame: np.ndarray, source: np.ndarray) -> np.ndarray:
        lock = float(np.clip(self.config.identity_lock, 0.0, 1.0))
        if lock <= 0:
            return frame

        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        source_mean, source_std = cv2.meanStdDev(source_lab)
        frame_mean, frame_std = cv2.meanStdDev(frame_lab)

        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        frame_mean = frame_mean.reshape(1, 1, 3)
        frame_std = frame_std.reshape(1, 1, 3)

        color_matched = (frame_lab - frame_mean) * (source_std / (frame_std + 1e-6)) + source_mean
        color_matched = np.clip(color_matched, 0, 255).astype(np.uint8)
        color_matched = cv2.cvtColor(color_matched, cv2.COLOR_LAB2BGR)

        return cv2.addWeighted(frame, 1.0 - lock, color_matched, lock, 0.0)

    def _apply_look(self, frame: np.ndarray, prev_stylized: np.ndarray | None) -> np.ndarray:
        if self.config.look == "natural" or self.config.look_strength <= 0:
            return frame

        detail = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        denoised = cv2.fastNlMeansDenoisingColored(detail, None, 5, 5, 7, 21)
        graded = self._filmic_curve(denoised)

        lab = cv2.cvtColor(graded, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        graded = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(graded, -1, sharpen_kernel)

        strength = float(np.clip(self.config.look_strength, 0.0, 1.0))
        stylized = cv2.addWeighted(frame, 1.0 - strength, sharpened, strength, 0.0)

        if prev_stylized is None:
            return stylized

        stabilization = float(np.clip(self.config.temporal_smooth, 0.0, 0.4))
        return cv2.addWeighted(stylized, 1.0 - stabilization, prev_stylized, stabilization, 0.0)

    def generate(self, source_image: Path, reference_video: Path, output_video: Path) -> None:
        src = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if src is None:
            raise FileNotFoundError(f"Could not load source image: {source_image}")

        ref_frames, ref_fps = self._read_reference_frames(reference_video, self.config.max_frames)
        h, w = ref_frames[0].shape[:2]
        src = self._resize_to_reference(src, (h, w))

        output_video.parent.mkdir(parents=True, exist_ok=True)
        fps = self.config.fps or ref_fps

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video}")

        generated = src.copy()
        prev_ref_gray = cv2.cvtColor(ref_frames[0], cv2.COLOR_BGR2GRAY)
        prev_stylized: np.ndarray | None = None

        first_frame = self._apply_look(generated, prev_stylized)
        writer.write(first_frame)
        prev_stylized = first_frame

        for i in tqdm(range(1, len(ref_frames)), desc="Transferring motion"):
            curr_ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY)

            H = self._estimate_global_transform(prev_ref_gray, curr_ref_gray)
            flow = self._dense_flow(prev_ref_gray, curr_ref_gray)

            moved = self._warp_with_homography(generated, H, self.config.global_scale)
            moved = self._warp_with_flow(moved, flow, self.config.flow_scale)

            mask = self._motion_focus_mask(flow, h, w)
            mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

            motion_mixed = (moved.astype(np.float32) * mask_3) + (generated.astype(np.float32) * (1.0 - mask_3))
            motion_mixed = np.clip(motion_mixed, 0, 255).astype(np.uint8)

            blended = cv2.addWeighted(
                motion_mixed,
                1.0 - self.config.blend_strength,
                src,
                self.config.blend_strength,
                0,
            )
            blended = self._apply_identity_lock(blended, src)

            temporal_smooth = float(np.clip(self.config.temporal_smooth, 0.0, 0.4))
            generated = cv2.addWeighted(blended, 1.0 - temporal_smooth, generated, temporal_smooth, 0.0)

            generated = cv2.bilateralFilter(generated, d=5, sigmaColor=20, sigmaSpace=8)
            stylized = self._apply_look(generated, prev_stylized)

            writer.write(stylized)
            prev_stylized = stylized
            prev_ref_gray = curr_ref_gray

        writer.release()


def validate_inputs(source: Path, reference: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source image file does not exist: {source}")
    if not reference.exists():
        raise FileNotFoundError(f"Reference video file does not exist: {reference}")

    source_as_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    reference_video = cv2.VideoCapture(str(reference))
    reference_is_video = reference_video.isOpened()
    reference_video.release()

    if source_as_image is None and reference_is_video:
        source_as_video = cv2.VideoCapture(str(source))
        source_is_video = source_as_video.isOpened()
        source_as_video.release()

        reference_as_image = cv2.imread(str(reference), cv2.IMREAD_COLOR)

        if source_is_video and reference_as_image is not None:
            raise ValueError(
                "Input types look swapped. --source must be an image and --reference must be a video. "
                f"You passed source='{source}' (video) and reference='{reference}' (image)."
            )

    if source_as_image is None:
        raise ValueError(
            f"Could not decode source image: {source}. Provide an image file like .jpg, .jpeg, or .png."
        )

    if not reference_is_video:
        raise ValueError(
            f"Could not decode reference video: {reference}. Provide a video file like .mp4, .mov, or .avi."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate motion-controlled video from source image and reference video."
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to source image")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference video")
    parser.add_argument("--output", type=Path, required=True, help="Path to output MP4")
    parser.add_argument("--flow-scale", type=float, default=0.8)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--blend-strength", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--look",
        choices=("natural", "cgi"),
        default="natural",
        help="Visual finishing preset. 'cgi' applies detail enhancement, filmic grading, and sharpening.",
    )
    parser.add_argument(
        "--look-strength",
        type=float,
        default=0.45,
        help="Strength for the selected --look preset from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--motion-focus",
        type=float,
        default=0.6,
        help="Upper-body motion emphasis from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--hand-boost",
        type=float,
        default=0.35,
        help="Additional emphasis for likely hand regions from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--temporal-smooth",
        type=float,
        default=0.2,
        help="Frame-to-frame stabilization strength from 0.0 to 0.4.",
    )
    parser.add_argument(
        "--identity-lock",
        type=float,
        default=0.45,
        help="Preserve source-character color identity from 0.0 to 1.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args.source, args.reference)
    cfg = MotionConfig(
        flow_scale=args.flow_scale,
        global_scale=args.global_scale,
        blend_strength=args.blend_strength,
        fps=args.fps,
        max_frames=args.max_frames,
        look=args.look,
        look_strength=args.look_strength,
        motion_focus=args.motion_focus,
        hand_boost=args.hand_boost,
        temporal_smooth=args.temporal_smooth,
        identity_lock=args.identity_lock,
    )

    generator = MotionControlGenerator(cfg)
    generator.generate(args.source, args.reference, args.output)
    print(f"Saved output: {args.output}")


if __name__ == "__main__":
    main()

