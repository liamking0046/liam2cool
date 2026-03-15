from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

@dataclass
class MotionConfig:
    flow_scale: float = 0.85
    global_scale: float = 1.0
    blend_strength: float = 0.12
    fps: float | None = None
    max_frames: int | None = None
    look: str = "natural"
    look_strength: float = 0.6
    motion_focus: float = 0.65
    hand_boost: float = 0.5
    temporal_smooth: float = 0.25
    identity_lock: float = 0.55
    structure_lock: float = 0.6
    lighting_transfer: float = 0.45
    flow_momentum: float = 0.5
    quality_passes: int = 2
    upperbody_stickiness: float = 0.8
    micro_motion: float = 0.4

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
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=6, blockSize=7)
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
        return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, pyr_scale=0.5, levels=5, winsize=25,
                                            iterations=5, poly_n=7, poly_sigma=1.5, flags=0)

    @staticmethod
    def _warp_with_flow(image: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0] * scale).astype(np.float32)
        map_y = (grid_y + flow[..., 1] * scale).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _iterative_flow_warp(self, image: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
        passes = max(1, int(self.config.quality_passes))
        step_scale = scale / passes
        warped = image
        for _ in range(passes):
            warped = self._warp_with_flow(warped, flow, step_scale)
        return warped

    @staticmethod
    def _upper_body_prior(height: int, width: int) -> np.ndarray:
        y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)
        x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        torso = np.exp(-((yy - 0.34) ** 2) / 0.09)
        hands_left = np.exp(-(((xx - 0.22) ** 2) + ((yy - 0.45) ** 2)) / 0.03)
        hands_right = np.exp(-(((xx - 0.78) ** 2) + ((yy - 0.45) ** 2)) / 0.03)
        prior = torso + 0.95 * (hands_left + hands_right)
        prior = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 5)
        return np.clip(prior / (np.max(prior) + 1e-6), 0.0, 1.0)

    def _motion_focus_mask(self, flow: np.ndarray, height: int, width: int) -> np.ndarray:
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        p95 = np.percentile(magnitude, 95)
        normalized = magnitude / (p95 + 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)
        prior = self._upper_body_prior(height, width)
        emphasized = normalized * (1.0 + self.config.motion_focus * prior + self.config.hand_boost * np.sqrt(prior))
        return cv2.GaussianBlur(np.clip(emphasized, 0.0, 1.0).astype(np.float32), (0, 0), 3)

    def generate(self, source_image: Path, reference_video: Path, output_video: Path) -> None:
        src = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if src is None:
            raise FileNotFoundError(f"Could not load source image: {source_image}")

        ref_frames, ref_fps = self._read_reference_frames(reference_video, self.config.max_frames)
        h, w = ref_frames[0].shape[:2]
        src = self._resize_to_reference(src, (h, w))

        output_video.parent.mkdir(parents=True, exist_ok=True)
        fps = self.config.fps or ref_fps
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video}")

        generated = src.copy()
        upper_mask = self._upper_body_prior(h, w)
        upper_mask3 = np.repeat(upper_mask[:, :, None], 3, axis=2)
        prev_ref_gray = cv2.cvtColor(ref_frames[0], cv2.COLOR_BGR2GRAY)
        prev_flow: np.ndarray | None = None

        writer.write(generated)

        for i in tqdm(range(1, len(ref_frames)), desc="Transferring motion"):
            ref_frame = ref_frames[i]
            curr_ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

            H = self._estimate_global_transform(prev_ref_gray, curr_ref_gray)
            flow = self._dense_flow(prev_ref_gray, curr_ref_gray)
            stabilized_flow = flow if prev_flow is None else 0.5 * flow + 0.5 * prev_flow

            moved = self._iterative_flow_warp(generated, stabilized_flow, self.config.flow_scale)

            motion_mask = self._motion_focus_mask(stabilized_flow, h, w)
            combined_mask = np.clip(motion_mask * 0.7 + upper_mask * 0.3, 0.0, 1.0)
            mask_3 = np.repeat(combined_mask[:, :, None], 3, axis=2)

            motion_mixed = (moved.astype(np.float32) * mask_3) + (generated.astype(np.float32) * (1.0 - mask_3))
            motion_mixed = np.clip(motion_mixed, 0, 255).astype(np.uint8)

            stick = float(np.clip(self.config.upperbody_stickiness, 0.0, 1.0))
            upper_driven = motion_mixed * upper_mask3 + src * (1.0 - upper_mask3) * stick
            upper_driven = np.clip(upper_driven, 0, 255).astype(np.uint8)

            temporal_smooth = float(np.clip(self.config.temporal_smooth, 0.0, 0.4))
            generated = cv2.addWeighted(upper_driven, 1.0 - temporal_smooth, generated, temporal_smooth, 0.0)

            generated = cv2.edgePreservingFilter(generated, flags=1, sigma_s=30, sigma_r=0.25)

            writer.write(generated)
            prev_ref_gray = curr_ref_gray
            prev_flow = stabilized_flow

        writer.release()
