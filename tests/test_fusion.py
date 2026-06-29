"""Tests for fuse_masks_to_3d (mask + geometry -> per-object 3D clouds).

These are pure-logic tests: no DA3, no SAM2, no GPU. We hand-build a tiny fake
Reconstruction + fake masks so we can check the *bookkeeping* — that points land
in the right bucket by mask_id, that an object seen in several frames has its
points piled together, and that the output dtypes match the contract.
"""

import numpy as np

from lumen3d.backbone import Reconstruction
from lumen3d.segmentation import ObjectMask
from lumen3d.fusion import fuse_masks_to_3d


def _fake_recon():
    """Two 2x2 frames. fx=fy=1, cx=cy=0, identity pose -> a pixel (u, v) at
    depth d unprojects to the round world point (u*d, v*d, d)."""
    # depth: every pixel at distance 2.
    depth = np.full((2, 2, 2), 2.0)

    # K: fx=fy=1, cx=cy=0 (so the math stays integer-clean).
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    intrinsics = np.array([K, K])

    # pose: identity for both frames -> world == camera.
    extrinsics = np.array([np.eye(4), np.eye(4)])

    # distinct colors per pixel per frame, so we can verify color alignment.
    images = np.array([
        [[[10, 10, 10], [20, 20, 20]],     # frame 0 row v=0
         [[30, 30, 30], [40, 40, 40]]],    # frame 0 row v=1
        [[[50, 50, 50], [60, 60, 60]],     # frame 1 row v=0
         [[70, 70, 70], [80, 80, 80]]],    # frame 1 row v=1
    ], dtype=np.uint8)

    conf = np.ones((2, 2, 2))

    # points/colors are required fields but unused by fuse_masks_to_3d.
    return Reconstruction(
        points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.uint8),
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        images=images,
        conf=conf,
    )


def _mask(true_at):
    """A 2x2 bool mask, True at the given (v, u) coordinates."""
    m = np.zeros((2, 2), dtype=bool)
    for v, u in true_at:
        m[v, u] = True
    return m


def test_groups_by_mask_id_and_piles_across_frames():
    # Arrange: object 7 appears in BOTH frames; object 12 only in frame 0.
    #   frame 0: obj 7 at pixel (v=1,u=0); obj 12 at pixel (v=0,u=0)
    #   frame 1: obj 7 at pixels (v=1,u=0) and (v=1,u=1)
    recon = _fake_recon()
    masks = [
        [ObjectMask(mask_id=7,  mask=_mask([(1, 0)])),
         ObjectMask(mask_id=12, mask=_mask([(0, 0)]))],
        [ObjectMask(mask_id=7,  mask=_mask([(1, 0), (1, 1)]))],
    ]

    # Act
    result = fuse_masks_to_3d(recon, masks, conf_thr=0.5)

    # Assert: both objects are present, keyed by masklet id.
    assert set(result.keys()) == {7, 12}

    pts7, cols7 = result[7]
    pts12, cols12 = result[12]

    # Object 7 piled across frames: 1 point (frame 0) + 2 points (frame 1) = 3.
    #   frame0 (v=1,u=0,d=2) -> (0, 2, 2)
    #   frame1 (v=1,u=0,d=2) -> (0, 2, 2);  (v=1,u=1,d=2) -> (2, 2, 2)
    assert pts7.shape == (3, 3)
    assert np.allclose(pts7, [[0, 2, 2],
                              [0, 2, 2],
                              [2, 2, 2]])
    # colors stay row-aligned with the points, in frame-then-scan order.
    assert np.array_equal(cols7, [[30, 30, 30],     # frame0 (1,0)
                                  [70, 70, 70],     # frame1 (1,0)
                                  [80, 80, 80]])    # frame1 (1,1)

    # Object 12 seen once: frame0 (v=0,u=0,d=2) -> (0, 0, 2).
    assert pts12.shape == (1, 3)
    assert np.allclose(pts12, [[0, 0, 2]])
    assert np.array_equal(cols12, [[10, 10, 10]])


def test_output_dtypes_match_the_contract():
    recon = _fake_recon()
    masks = [
        [ObjectMask(mask_id=1, mask=_mask([(0, 0)]))],
        [ObjectMask(mask_id=1, mask=_mask([(0, 0)]))],
    ]

    result = fuse_masks_to_3d(recon, masks, conf_thr=0.5)
    pts, cols = result[1]

    assert pts.dtype == np.float32
    assert cols.dtype == np.uint8
