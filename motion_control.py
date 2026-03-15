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

        writer.write(generated)

        for i in tqdm(range(1, len(ref_frames)), desc="Transferring motion"):
            curr_ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY)

            H = self._estimate_global_transform(prev_ref_gray, curr_ref_gray)
            flow = self._dense_flow(prev_ref_gray, curr_ref_gray)

            moved = self._warp_with_homography(generated, H, self.config.global_scale)
            moved = self._warp_with_flow(moved, flow, self.config.flow_scale)

            generated = cv2.addWeighted(
                moved,
                1.0 - self.config.blend_strength,
                src,
                self.config.blend_strength,
                0,
            )

            generated = cv2.bilateralFilter(generated, d=5, sigmaColor=25, sigmaSpace=10)
            writer.write(generated)
            prev_ref_gray = curr_ref_gray

        writer.release()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MotionConfig(
        flow_scale=args.flow_scale,
        global_scale=args.global_scale,
        blend_strength=args.blend_strength,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    generator = MotionControlGenerator(cfg)
    generator.generate(args.source, args.reference, args.output)
    print(f"Saved output: {args.output}")


if __name__ == "__main__":
    main()
