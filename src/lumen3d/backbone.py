from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Reconstruction:
    """A reconstructed point cloud: 3D geometry of a scene.

    `points` and `colors` are always present (the merged cloud). The remaining
    fields are the per-frame arrays needed to unproject a single mask back to 3D
    (mask -> 3D). They are optional: present when the backbone exposes them
    (DA3 does), otherwise `None`.

    Attributes:
        points: Float array of shape (N, 3) — the XYZ position of each point.
        colors: Uint8 array of shape (N, 3) — the RGB color of each point,
            row-aligned with `points` (colors[i] is the color of points[i]).
        depth: Float array of shape (F, H, W) — per-frame depth map;
            depth[i][v, u] is how far pixel (u, v) of frame i is.
        intrinsics: Float array of shape (F, 3, 3) — per-frame camera matrix K
            (focal lengths fx, fy and principal point cx, cy).
        extrinsics: Per-frame camera pose, world->camera (w2c). ⚠️ Must be
            inverted to camera->world (and homogenized to 4x4) before
            unprojecting — see geometry.to_homogeneous / unproject.
        images: Uint8 array of shape (F, H, W, 3) — the processed RGB frames
            (DA3's processed_images).
        conf: Float array of shape (F, H, W) — per-pixel confidence; low values
            are filtered out before a point is created.
    """
    points: np.ndarray
    colors: np.ndarray
    depth: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    images: np.ndarray | None = None
    conf: np.ndarray | None = None


class Backbone(ABC):
    """Contract for any reconstruction model (DA3, VGGT, π³).

    Implementations turn a list of frame paths into a 3D point cloud.
    """

    @abstractmethod
    def reconstruct(self, frames: list[Path]) -> Reconstruction:
        ...

class DA3Backbone(Backbone):

    def __init__(self, weights="depth-anything/DA3-LARGE", device=None):
        self.weights = weights   
        self.device = device     
        self._model = None

    def _load_model(self):
        if self._model is None:            # not loaded yet?
            from depth_anything_3.api import DepthAnything3   # ← lazy import, here not at top
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self._model = DepthAnything3.from_pretrained(self.weights).to(device)
        return self._model

    def reconstruct(self, frames: list[Path]) -> Reconstruction:
        model = self._load_model()

        paths = [str(p) for p in frames] 

        pred = model.inference(image=paths, export_dir=None)

        from depth_anything_3.utils.export.glb import (
        _depths_to_world_points_with_colors, get_conf_thresh,
    )

        conf_thr = get_conf_thresh(pred, getattr(pred, "sky_mask", None), 1.05, 40.0)   # (e)
        points, colors = _depths_to_world_points_with_colors(   # (f) unproject -> arrays
            pred.depth, pred.intrinsics, pred.extrinsics,
            pred.processed_images, pred.conf, conf_thr,
        )

        return Reconstruction(
            points=points,
            colors=colors,
            depth=pred.depth,
            intrinsics=pred.intrinsics,
            extrinsics=pred.extrinsics,
            images=pred.processed_images,
            conf=pred.conf,
        )
