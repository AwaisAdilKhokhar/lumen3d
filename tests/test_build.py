"""Tests for build_scene (the produce-side front door).

build_scene owns no model math — it just runs the machines in order and packs
their outputs into a bundle. So we fake the three GPU machines (backbone,
segmenter, embedder) with stand-ins that return canned data, and let the cheap
pure-NumPy machine, associate_masks_3d, run for real. The tests then check the
*wiring*: right pieces in the right compartments, geometry and embeddings keyed
by the SAME instance ids, the low-res images handed to the embedder, and
save_path round-tripping through cache.load_scene.

Note on ids: the fake segmenter returns detections tagged 7 and 12, but those
are raw per-detection ids. build_scene runs the real 3D association, which
relabels them to instance ids (0, 1 here) — so the assertions below are written
against the instance ids, not the segmenter's raw ones.
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
    overlap -> the associator merges it into a single instance), plus a second
    object at (0,0) seen only in frame 0. The raw ids (7, 12) are meaningless;
    association reassigns instance ids 0 and 1."""
    return [
        [ObjectMask(mask_id=7,  mask=_mask([(1, 0)])),
         ObjectMask(mask_id=12, mask=_mask([(0, 0)]))],
        [ObjectMask(mask_id=7,  mask=_mask([(1, 0), (1, 1)]))],
    ]


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
    canned_embeddings = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
    }
    backbone = _FakeBackbone(recon)
    segmenter = _FakeSegmenter(masks)
    embedder = _FakeEmbedder(canned_embeddings)

    # Act (min_points=1: the fake masks are only a pixel or two)
    bundle = build_scene(
        ["a.jpg", "b.jpg"], backbone, segmenter, embedder,
        conf_thr=0.5, min_points=1,
    )

    # Assert: the box has exactly the three compartments.
    assert set(bundle.keys()) == {"embeddings", "geometry", "scene"}

    # embeddings compartment is what the embedder produced, untouched.
    assert bundle["embeddings"] is canned_embeddings

    # geometry compartment is the REAL association output: the merged object
    # (instance 0) piled across both frames (3 points), the frame-0-only object
    # (instance 1) seen once (1 point).
    geometry = bundle["geometry"]
    assert set(geometry.keys()) == {0, 1}
    assert geometry[0][0].shape == (3, 3)
    assert geometry[1][0].shape == (1, 3)

    # geometry and embeddings must be keyed by the SAME instance ids, and those
    # are exactly the ids the embedder was handed on the relabeled masks.
    seen_ids = {obj.mask_id for frame in embedder.masks_seen for obj in frame}
    assert seen_ids == set(geometry.keys())

    # scene compartment is the reconstruction's full cloud.
    scene_points, scene_colors = bundle["scene"]
    assert np.array_equal(scene_points, recon.points)
    assert np.array_equal(scene_colors, recon.colors)


def test_build_scene_embeds_the_reconstruction_images():
    # The decision we locked: meaning-numbers come from DA3's (low-res) images,
    # not the original frames. Prove build_scene actually hands recon.images to
    # the embedder.
    recon = _fake_recon()
    embedder = _FakeEmbedder({7: np.zeros(2, dtype=np.float32)})

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
    canned_embeddings = {7: np.array([1.0, 0.0], dtype=np.float32)}
    out = tmp_path / "scene.pkl"

    bundle = build_scene(
        ["a.jpg", "b.jpg"],
        _FakeBackbone(recon),
        _FakeSegmenter(_fake_masks()),
        _FakeEmbedder(canned_embeddings),
        save_path=out,
    )

    assert out.exists()
    loaded = load_scene(out)
    assert set(loaded.keys()) == {"embeddings", "geometry", "scene"}
    assert np.array_equal(loaded["embeddings"][7], bundle["embeddings"][7])
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
