"""Tests for associate_masks_3d — instance identity from geometry AND meaning.

The associator is pure NumPy, so these run with no GPU and no models: we hand it
a tiny Reconstruction plus a dict of fake per-detection embeddings. The camera is
clean (fx=fy=1, cx=cy=0, identity pose), so pixel (u, v) at depth d lands at
world (u*d, v*d, d) and we can predict exactly what should merge.
"""

import numpy as np

from lumen3d.backbone import Reconstruction
from lumen3d.segmentation import ObjectMask
from lumen3d.association import associate_masks_3d, _voxel_set, _voxel_iou, _cosine


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


def _same_embs(masks):
    """One identical embedding per detection -> the semantic gate always passes,
    so merging is decided by geometry alone (used to test the geometric logic)."""
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return {obj.mask_id: vec for frame in masks for obj in frame}


def test_same_object_in_two_frames_merges_into_one_instance():
    recon = _recon(n_frames=2)
    # The identical 3x3 patch in both frames -> identical world points -> merge.
    masks = [
        [ObjectMask(mask_id=100, mask=_block((0, 3), (0, 3)))],
        [ObjectMask(mask_id=200, mask=_block((0, 3), (0, 3)))],
    ]

    relabeled, geometry, embeddings = associate_masks_3d(
        recon, masks, _same_embs(masks), min_points=1)

    # One instance, its points pooled from both frames (9 + 9).
    assert len(geometry) == 1
    (inst_id,) = geometry.keys()
    assert geometry[inst_id][0].shape == (18, 3)
    assert set(embeddings.keys()) == {inst_id}

    # Both frames' relabeled masks carry that one instance id (frame-stable).
    assert [m.mask_id for m in relabeled[0]] == [inst_id]
    assert [m.mask_id for m in relabeled[1]] == [inst_id]


def test_semantic_gate_keeps_unlike_overlapping_objects_apart():
    # THE FIX: two detections that overlap in space (same patch) but whose
    # embeddings point in different directions must NOT merge — this is the
    # door-next-to-a-trash-can case geometry alone got wrong.
    recon = _recon(n_frames=2)
    masks = [
        [ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3)))],
        [ObjectMask(mask_id=2, mask=_block((0, 3), (0, 3)))],
    ]
    embs = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),   # orthogonal -> cosine 0
    }

    _, geometry, _ = associate_masks_3d(recon, masks, embs, min_points=1, sim_thr=0.85)

    assert len(geometry) == 2          # meaning split what geometry would have merged


def test_similar_overlapping_objects_still_merge():
    # Same overlap, but near-parallel embeddings (cosine ~0.997) -> they merge,
    # confirming the gate isn't just refusing everything.
    recon = _recon(n_frames=2)
    masks = [
        [ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3)))],
        [ObjectMask(mask_id=2, mask=_block((0, 3), (0, 3)))],
    ]
    embs = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 0.08], dtype=np.float32),
    }

    _, geometry, _ = associate_masks_3d(recon, masks, embs, min_points=1, sim_thr=0.85)

    assert len(geometry) == 1


def test_two_separated_objects_stay_distinct():
    recon = _recon(n_frames=1)
    masks = [[
        ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),      # left
        ObjectMask(mask_id=2, mask=_block((0, 3), (7, 10))),     # far right
    ]]

    relabeled, geometry, _ = associate_masks_3d(
        recon, masks, _same_embs(masks), min_points=1)

    # No spatial overlap -> two instances even though the embeddings match.
    assert len(geometry) == 2
    ids = [m.mask_id for m in relabeled[0]]
    assert len(set(ids)) == 2


def test_min_points_drops_tiny_detections():
    recon = _recon(n_frames=1)
    masks = [[
        ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),      # 9 pixels, kept
        ObjectMask(mask_id=2, mask=_block((5, 6), (5, 6))),      # 1 pixel, dropped
    ]]

    relabeled, geometry, _ = associate_masks_3d(
        recon, masks, _same_embs(masks), min_points=5)

    assert len(geometry) == 1
    assert len(relabeled[0]) == 1          # the tiny detection is gone


def test_detection_without_an_embedding_is_dropped():
    recon = _recon(n_frames=1)
    masks = [[
        ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),
        ObjectMask(mask_id=2, mask=_block((0, 3), (7, 10))),
    ]]
    embs = {1: np.array([1.0, 0.0], dtype=np.float32)}   # id 2 has no embedding

    relabeled, geometry, embeddings = associate_masks_3d(
        recon, masks, embs, min_points=1)

    assert len(geometry) == 1
    assert len(relabeled[0]) == 1
    assert set(embeddings.keys()) == set(geometry.keys())


def test_geometry_embeddings_and_masks_share_the_same_ids():
    recon = _recon(n_frames=2)
    masks = [
        [ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3))),
         ObjectMask(mask_id=2, mask=_block((0, 3), (7, 10)))],
        [ObjectMask(mask_id=3, mask=_block((0, 3), (0, 3)))],
    ]

    relabeled, geometry, embeddings = associate_masks_3d(
        recon, masks, _same_embs(masks), min_points=1)

    seen = {m.mask_id for frame in relabeled for m in frame}
    assert seen == set(geometry.keys()) == set(embeddings.keys())
    # ids are a dense range starting at 0.
    assert set(geometry.keys()) == set(range(len(geometry)))


def test_instance_embedding_is_the_mean_of_its_members():
    recon = _recon(n_frames=2)
    masks = [
        [ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3)))],
        [ObjectMask(mask_id=2, mask=_block((0, 3), (0, 3)))],
    ]
    embs = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 0.02], dtype=np.float32),   # close enough to merge
    }

    _, geometry, embeddings = associate_masks_3d(recon, masks, embs, min_points=1, sim_thr=0.85)

    assert len(geometry) == 1
    (inst_id,) = embeddings.keys()
    assert np.allclose(embeddings[inst_id], [1.0, 0.01])


def test_invalid_pixels_are_excluded_from_fragments():
    recon = _recon(n_frames=1)
    recon.depth[0][0, 0] = -1.0            # one pixel of the patch has bad depth
    masks = [[ObjectMask(mask_id=1, mask=_block((0, 3), (0, 3)))]]

    _, geometry, _ = associate_masks_3d(recon, masks, _same_embs(masks), min_points=1)

    # 9 pixels in the patch, but the invalid one is dropped -> 8 points.
    (inst_id,) = geometry.keys()
    assert geometry[inst_id][0].shape == (8, 3)


# --- helpers ---------------------------------------------------------------

def test_voxel_iou_of_a_set_with_itself_is_one():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    a = _voxel_set(pts, voxel_size=0.5)
    assert _voxel_iou(a, a) == 1.0


def test_voxel_iou_is_zero_for_disjoint_clouds():
    a = _voxel_set(np.array([[0.0, 0.0, 0.0]]), voxel_size=0.5)
    b = _voxel_set(np.array([[100.0, 100.0, 100.0]]), voxel_size=0.5)
    assert _voxel_iou(a, b) == 0.0


def test_cosine_basic_cases():
    assert _cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
    assert _cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert _cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0
