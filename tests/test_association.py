"""Tests for associate_masks_3d — geometry-based instance identity (Option B).

The associator is pure NumPy, so these run with no GPU and no models. We build a
tiny Reconstruction with a clean camera (fx=fy=1, cx=cy=0, identity pose), where
pixel (u, v) at depth d lands at world (u*d, v*d, d). That lets us place fake
detections at known world locations and predict exactly what should merge.
"""

import numpy as np

from lumen3d.backbone import Reconstruction
from lumen3d.segmentation import ObjectMask
from lumen3d.association import associate_masks_3d, _voxel_set, _voxel_iou


H = W = 10


def _recon(n_frames=2, depth_val=1.0):
    depth = np.full((n_frames, H, W), depth_val)
    K = np.eye(3)
    intrinsics = np.array([K] * n_frames)
    extrinsics = np.array([np.eye(4)] * n_frames)
    images = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    conf = np.ones((n_frames, H, W))
    # A two-corner "cloud" just to set the scene scale (voxel size is a fraction
    # of the bounding-box diagonal).
    points = np.array([[0, 0, depth_val], [W - 1, H - 1, depth_val]], dtype=np.float32)
    colors = np.zeros((2, 3), dtype=np.uint8)
    return Reconstruction(points, colors, depth, intrinsics, extrinsics, images, conf)


def _block(rows, cols):
    m = np.zeros((H, W), dtype=bool)
    m[rows[0]:rows[1], cols[0]:cols[1]] = True
    return m


def test_same_object_in_two_frames_merges_into_one_instance():
    recon = _recon(n_frames=2)
    # The identical 3x3 patch in both frames -> identical world points -> merge.
    masks = [
        [ObjectMask(mask_id=100, mask=_block((0, 3), (0, 3)))],
        [ObjectMask(mask_id=200, mask=_block((0, 3), (0, 3)))],
    ]

    relabeled, geometry = associate_masks_3d(recon, masks, min_points=1)

    # One instance, its points pooled from both frames (9 + 9).
    assert len(geometry) == 1
    (inst_id,) = geometry.keys()
    assert geometry[inst_id][0].shape == (18, 3)

    # Both frames' relabeled masks carry that one instance id (identity is now
    # frame-stable).
    assert [m.mask_id for m in relabeled[0]] == [inst_id]
    assert [m.mask_id for m in relabeled[1]] == [inst_id]


def test_two_separated_objects_stay_distinct():
    recon = _recon(n_frames=1)
    masks = [[
        ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),      # left
        ObjectMask(mask_id=2, mask=_block((0, 3), (7, 10))),     # far right
    ]]

    relabeled, geometry = associate_masks_3d(recon, masks, min_points=1)

    # No spatial overlap -> two instances with different ids.
    assert len(geometry) == 2
    ids = [m.mask_id for m in relabeled[0]]
    assert len(set(ids)) == 2


def test_min_points_drops_tiny_detections():
    recon = _recon(n_frames=1)
    masks = [[
        ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),      # 9 pixels, kept
        ObjectMask(mask_id=2, mask=_block((5, 6), (5, 6))),      # 1 pixel, dropped
    ]]

    relabeled, geometry = associate_masks_3d(recon, masks, min_points=5)

    assert len(geometry) == 1
    assert len(relabeled[0]) == 1          # the tiny detection is gone


def test_geometry_and_relabeled_masks_share_the_same_ids():
    recon = _recon(n_frames=2)
    masks = [
        [ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),
         ObjectMask(mask_id=2, mask=_block((0, 3), (7, 10)))],
        [ObjectMask(mask_id=3, mask=_block((0, 3), (0, 3)))],
    ]

    relabeled, geometry = associate_masks_3d(recon, masks, min_points=1)

    seen = {m.mask_id for frame in relabeled for m in frame}
    assert seen == set(geometry.keys())
    # geometry ids are a dense range starting at 0.
    assert set(geometry.keys()) == set(range(len(geometry)))


def test_invalid_pixels_are_excluded_from_fragments():
    recon = _recon(n_frames=1)
    recon.depth[0][0, 0] = -1.0            # one pixel of the patch has bad depth
    masks = [[ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3)))]]

    _, geometry = associate_masks_3d(recon, masks, min_points=1)

    # 9 pixels in the patch, but the invalid one is dropped -> 8 points.
    (inst_id,) = geometry.keys()
    assert geometry[inst_id][0].shape == (8, 3)


# --- voxel helpers ---------------------------------------------------------

def test_voxel_iou_of_a_set_with_itself_is_one():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    a = _voxel_set(pts, voxel_size=0.5)
    assert _voxel_iou(a, a) == 1.0


def test_voxel_iou_is_zero_for_disjoint_clouds():
    a = _voxel_set(np.array([[0.0, 0.0, 0.0]]), voxel_size=0.5)
    b = _voxel_set(np.array([[100.0, 100.0, 100.0]]), voxel_size=0.5)
    assert _voxel_iou(a, b) == 0.0
