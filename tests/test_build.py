"""Tests for build_scene (the produce-side front door).

build_scene owns no model math — it just runs the machines in order and packs
their outputs into a bundle. So we fake the three GPU machines (backbone,
segmenter, embedder) with stand-ins that return canned data, and let the cheap
pure-NumPy machine, associate_masks_3d, run for real. The tests then check the
*wiring*: right pieces in the right compartments, geometry and embeddings keyed
by the SAME instance ids, the low-res images handed to the embedder, and
save_path round-tripping through cache.load_scene.

Note on ids: the fake segmenter returns detections with unique raw ids (7, 8,
12). build_scene embeds each detection, then runs the real 3D association, which
relabels everything to instance ids (0, 1 here) — so the assertions below are
written against the instance ids, not the segmenter's raw ones.
"""

import numpy as np

from lumen3d.backbone import Reconstruction
from lumen3d.segmentation import ObjectMask
from lumen3d.build import build_scene
from lumen3d.cache import load_scene


def _fake_recon():
    """Two 2x2 frames, same clean camera as test_fusion: fx=fy=1, cx=cy=0,
    identity pose -> pixel (u, v) at depth d lands at world (u*d, v*d, d).
    Also carries a tiny full cloud in points/colors so we can check `scene`."""
    depth = np.full((2, 2, 2), 2.0)
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    intrinsics = np.array([K, K])
    extrinsics = np.array([np.eye(4), np.eye(4)])
    images = np.array([
        [[[10, 10, 10], [20, 20, 20]],
         [[30, 30, 30], [40, 40, 40]]],
        [[[50, 50, 50], [60, 60, 60]],
         [[70, 70, 70], [80, 80, 80]]],
    ], dtype=np.uint8)
    conf = np.ones((2, 2, 2))
    return Reconstruction(
        points=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        colors=np.array([[11, 22, 33], [44, 55, 66]], dtype=np.uint8),
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        images=images,
        conf=conf,
    )


def _mask(true_at):
    m = np.zeros((2, 2), dtype=bool)
    for v, u in true_at:
        m[v, u] = True
    return m


def _fake_masks():
    """One object at pixel column 0's bottom, seen in both frames (its 3D points
    overlap -> the associator can merge it into a single instance), plus a second
    object at (0,0) seen only in frame 0. Every detection has a UNIQUE raw id
    (7, 12, 8) like real segment() output; association reassigns instance ids."""
    return [
        [ObjectMask(mask_id=7,  mask=_mask([(1, 0)])),
         ObjectMask(mask_id=12, mask=_mask([(0, 0)]))],
        [ObjectMask(mask_id=8,  mask=_mask([(1, 0), (1, 1)]))],
    ]


def _det_embeddings():
    """One embedding per detection, keyed by raw id. The two overlapping
    detections (7, 8) share a vector so they merge; 12 differs (and is spatially
    separate anyway)."""
    return {
        7:  np.array([1.0, 0.0], dtype=np.float32),
        8:  np.array([1.0, 0.0], dtype=np.float32),
        12: np.array([0.0, 1.0], dtype=np.float32),
    }


class _FakeBackbone:
    """reconstruct() ignores the frames and returns a fixed Reconstruction."""

    def __init__(self, recon):
        self._recon = recon
        self.frames_seen = None

    def reconstruct(self, frames):
        self.frames_seen = frames
        return self._recon


class _FakeSegmenter:
    """segment() ignores the frames and returns canned per-frame masks."""

    def __init__(self, masks):
        self._masks = masks

    def segment(self, frames):
        return self._masks


class _FakeEmbedder:
    """embed_regions() returns a canned dict and records the images it was
    handed, so a test can assert it received the reconstruction's images."""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.images_seen = None
        self.masks_seen = None

    def embed_regions(self, images, masks):
        self.images_seen = images
        self.masks_seen = masks
        return self._embeddings


def test_build_scene_assembles_the_bundle():
    # Arrange
    recon = _fake_recon()
    masks = _fake_masks()
    backbone = _FakeBackbone(recon)
    segmenter = _FakeSegmenter(masks)
    embedder = _FakeEmbedder(_det_embeddings())

    # Act (min_points=1: the fake masks are only a pixel or two)
    bundle = build_scene(
        ["a.jpg", "b.jpg"], backbone, segmenter, embedder,
        conf_thr=0.5, min_points=1,
    )

    # Assert: the box has exactly the three compartments.
    assert set(bundle.keys()) == {"embeddings", "geometry", "scene"}

    # geometry compartment is the REAL association output: the merged object
    # (instance 0) piled across both frames (3 points), the frame-0-only object
    # (instance 1) seen once (1 point).
    geometry = bundle["geometry"]
    assert set(geometry.keys()) == {0, 1}
    assert geometry[0][0].shape == (3, 3)
    assert geometry[1][0].shape == (1, 3)

    # embeddings come out of association (mean of member detections) and MUST be
    # keyed by the same instance ids as geometry.
    embeddings = bundle["embeddings"]
    assert set(embeddings.keys()) == set(geometry.keys())
    assert np.allclose(embeddings[0], [1.0, 0.0])   # mean of the two merged dets
    assert np.allclose(embeddings[1], [0.0, 1.0])

    # scene compartment is the reconstruction's full cloud.
    scene_points, scene_colors = bundle["scene"]
    assert np.array_equal(scene_points, recon.points)
    assert np.array_equal(scene_colors, recon.colors)


def test_build_scene_embeds_the_reconstruction_images():
    # The decision we locked: meaning-numbers come from DA3's (low-res) images,
    # not the original frames. Prove build_scene actually hands recon.images to
    # the embedder.
    recon = _fake_recon()
    embedder = _FakeEmbedder(_det_embeddings())

    build_scene(
        ["a.jpg", "b.jpg"],
        _FakeBackbone(recon),
        _FakeSegmenter(_fake_masks()),
        embedder,
        min_points=1,
    )

    assert embedder.images_seen is recon.images


def test_build_scene_saves_when_path_given(tmp_path):
    # With save_path set, the same bundle must be readable back off disk.
    recon = _fake_recon()
    out = tmp_path / "scene.pkl"

    bundle = build_scene(
        ["a.jpg", "b.jpg"],
        _FakeBackbone(recon),
        _FakeSegmenter(_fake_masks()),
        _FakeEmbedder(_det_embeddings()),
        min_points=1,
        save_path=out,
    )

    assert out.exists()
    loaded = load_scene(out)
    assert set(loaded.keys()) == {"embeddings", "geometry", "scene"}
    assert np.array_equal(loaded["embeddings"][0], bundle["embeddings"][0])
    assert np.array_equal(loaded["scene"][0], bundle["scene"][0])


def test_build_scene_does_not_write_without_a_path(tmp_path):
    # Default save_path=None must leave the disk untouched.
    build_scene(
        ["a.jpg", "b.jpg"],
        _FakeBackbone(_fake_recon()),
        _FakeSegmenter(_fake_masks()),
        _FakeEmbedder({7: np.zeros(2, dtype=np.float32)}),
    )

    assert list(tmp_path.iterdir()) == []
