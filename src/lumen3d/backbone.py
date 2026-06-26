from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Reconstruction:
    """A reconstructed point cloud: 3D geometry of a scene.

    Attributes:
        points: Float array of shape (N, 3) — the XYZ position of each point.
        colors: Uint8 array of shape (N, 3) — the RGB color of each point,
            row-aligned with `points` (colors[i] is the color of points[i]).
    """
    points: np.ndarray
    colors: np.ndarray


class Backbone(ABC):
    """Contract for any reconstruction model (DA3, VGGT, π³).

    Implementations turn a list of frame paths into a 3D point cloud.
    """

    @abstractmethod
    def reconstruct(self, frames: list[Path]) -> Reconstruction:
        ...

class DA3Backbone(Backbone):

    def reconstruct(self, frames: list[Path]) -> Reconstruction:
         raise NotImplementedError('I havent implemented the reconstruct method in DA3BackBone yet')




