"""this module defines the segmentation contract and the SAM2 implementation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np


@dataclass
class ObjectMask:
    """A mask of a single object which would include the mask array (boolean) and the object id

    Attributes:
        mask_id: int  — the id of a particular object which remains consistent across frames
        mask: bool array of shape (H, W) — Each mask isolates a single specific object
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
    """SAM2 segmenter (Meta). Loads weights lazily on first segment() call."""

    def __init__(self, weights="facebook/sam2-hiera-large", device=None):
        self.weights = weights   
        self.device = device     
        self._tracker = None
        self._finder = None

    def _load_model(self):
        if self._tracker is None or self._finder is None:            # not loaded yet?
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # ← lazy import, here not at top
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self._tracker = SAM2VideoPredictor.from_pretrained(self.weights)
            self._finder  = SAM2AutomaticMaskGenerator.from_pretrained(self.weights)
        return self._tracker,self._finder

    

    def segment(self, frames: list[Path]) -> list[ list[ObjectMask] ]:

        tracker, finder = self._load_model()

        # --- DISCOVER: Finder outlines every object on frame 0 ---
        bgr = cv2.imread(str(frames[0]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = finder.generate(rgb)

        # --- HANDOFF (setup): give the whole video to the Tracker ---
        # The Tracker's init_state demands EITHER an .mp4 OR a folder of JPEGs named as
        # bare zero-padded integers (00000.jpg, 00001.jpg, ...), because internally it
        # sorts with int(filename). Our frames are named frame_00000.jpg, which would
        # crash that int(). So we ADAPT: copy each frame into a temp folder under the
        # name SAM2 expects, then hand that folder over.
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame_path in enumerate(frames):
                shutil.copy(frame_path, Path(tmpdir) / f"{i:05d}.jpg")
            state = tracker.init_state(tmpdir)
            for i, m in enumerate(masks):
                tracker.add_new_mask(state, frame_idx=0, obj_id=i, mask=m["segmentation"])

            results = []                                    # the list[list[ObjectMask]] we'll return
            for frame_idx, obj_ids, video_res_masks in tracker.propagate_in_video(state):
                frame_masks = []                            # the objects in THIS one frame
                for j, obj_id in enumerate(obj_ids):
                    bool_mask = (video_res_masks[j, 0] > 0.0).cpu().numpy()
                    frame_masks.append(ObjectMask(mask_id=int(obj_id), mask=bool_mask))
                results.append(frame_masks)
            return results

            

        