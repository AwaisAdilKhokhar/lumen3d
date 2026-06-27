



from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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
        self._model = None

    

    def segment(self, frames: list[Path]) -> list[ list[ObjectMask] ]:

        raise NotImplementedError('Its supposed to be completed in phase B')
        

        