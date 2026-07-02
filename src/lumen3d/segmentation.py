"""this module defines the segmentation contract and the SAM2 implementation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class ObjectMask:
    """A single object outline on a single frame: a boolean mask plus an id.

    Attributes:
        mask_id: int — which object this mask belongs to.
            ⚠️ The MEANING depends on the stage:
              - Straight out of `Segmenter.segment`, ids are per-DETECTION and
                carry NO cross-frame identity (the finder runs on each frame
                independently, so the same real object gets a different id in
                each frame it appears in).
              - After `association.associate_masks_3d`, ids are per-INSTANCE and
                DO stay consistent across frames (same id == same object in 3D).
            Fusion and embedding both group by `mask_id`, so they must be fed the
            associated (instance-labeled) masks, not the raw detections.
        mask: bool array of shape (H, W) — True where this object is.
    """
    mask_id: int
    mask: np.ndarray


class Segmenter(ABC):
    """Contract for any segmentation model like SAM2

    Implementations turn a list of frame paths into a list of ObjectMasks per frame
    """

    @abstractmethod
    def segment(self, frames: list[Path]) -> list[ list[ObjectMask] ]:
        ...


class SAM2Segmenter(Segmenter):
    """SAM2 segmenter (Meta). Loads weights lazily on first segment() call.

    Discovery-per-frame: the automatic mask generator ("Finder") runs on EVERY
    frame, so objects that only appear later in the clip get found too. It does
    NOT establish cross-frame identity — that is `association.associate_masks_3d`'s
    job, done in 3D. (The old design ran the Finder on frame 0 only and had the
    SAM2 video Tracker propagate those masks forward, which meant anything absent
    from frame 0 was invisible to the whole pipeline.)
    """

    def __init__(self, weights="facebook/sam2-hiera-large", device=None):
        self.weights = weights
        self.device = device
        self._finder = None

    def _load_model(self):
        if self._finder is None:            # not loaded yet?
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # ← lazy import, here not at top
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self._finder = SAM2AutomaticMaskGenerator.from_pretrained(self.weights)
        return self._finder

    def segment(self, frames: list[Path]) -> list[ list[ObjectMask] ]:
        finder = self._load_model()

        results = []          # the list[list[ObjectMask]] we'll return
        next_id = 0           # a running id so every raw detection is distinct
        for frame_path in frames:
            bgr = cv2.imread(str(frame_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            masks = finder.generate(rgb)          # DISCOVER: outline every object on THIS frame

            frame_masks = []
            for m in masks:
                # Ids are per-detection here (no cross-frame meaning yet) — see
                # ObjectMask.mask_id. associate_masks_3d assigns the real,
                # frame-stable instance ids afterwards.
                frame_masks.append(ObjectMask(mask_id=next_id, mask=m["segmentation"]))
                next_id += 1
            results.append(frame_masks)
        return results
