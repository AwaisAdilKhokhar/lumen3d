"""Tests for SigLIPEmbedder.embed_regions (crop + embed + average per object).

These are pure-logic tests: no SigLIP, no transformers, no GPU. We replace the
one model-touching method, `_embed_image`, with a fake that returns the crop's
top-left pixel as a 3-vector. Because that pixel is the bounding box's top-left
corner, the fake lets us verify *all* the bookkeeping at once: that the bounding
box is computed correctly, that vectors land in the right bucket by mask_id, that
an object seen in several frames has its vectors AVERAGED, that an empty mask is
skipped, and that the output dtype matches the contract.
"""

import numpy as np

from lumen3d.embedding import SigLIPEmbedder
from lumen3d.segmentation import ObjectMask


def _fake_images():
    """Two 4x4 RGB frames. Each pixel (v, u) is the gray value (v*4 + u),
    so every pixel is unique and a crop's top-left pixel reveals (ymin, xmin)."""
    frames = []
    for _ in range(2):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for v in range(4):
            for u in range(4):
                frame[v, u] = v * 4 + u
        frames.append(frame)
    return np.array(frames)


def _mask(true_at):
    """A 4x4 bool mask, True at the given (v, u) coordinates."""
    m = np.zeros((4, 4), dtype=bool)
    for v, u in true_at:
        m[v, u] = True
    return m


def _embedder_with_fake():
    """A SigLIPEmbedder whose model call is replaced by 'return the crop's
    top-left pixel as a float vector' — never loads SigLIP."""
    emb = SigLIPEmbedder()
    emb._embed_image = lambda image: image[0, 0].astype(np.float32)
    return emb


def test_groups_by_mask_id_and_averages_across_frames():
    # Arrange: object 7 appears in BOTH frames (different boxes); object 12 only
    # in frame 0; object 99 is an all-False mask that must be skipped.
    #   frame 0: obj 7 at (v=1,u=1) -> box top-left (1,1) -> pixel 5  -> [5,5,5]
    #            obj 12 at (v=0,u=3) -> box top-left (0,3) -> pixel 3 -> [3,3,3]
    #   frame 1: obj 7 at (v=2,u=0),(v=2,u=2) -> box top-left (2,0) -> pixel 8 -> [8,8,8]
    #            obj 99 empty -> skipped
    images = _fake_images()
    masks = [
        [ObjectMask(mask_id=7,  mask=_mask([(1, 1)])),
         ObjectMask(mask_id=12, mask=_mask([(0, 3)]))],
        [ObjectMask(mask_id=7,  mask=_mask([(2, 0), (2, 2)])),
         ObjectMask(mask_id=99, mask=_mask([]))],
    ]

    # Act
    result = _embedder_with_fake().embed_regions(images, masks)

    # Assert: only the two non-empty objects are present, keyed by masklet id.
    assert set(result.keys()) == {7, 12}

    # Object 7 averaged across frames: ([5,5,5] + [8,8,8]) / 2 = [6.5,6.5,6.5].
    assert np.allclose(result[7], [6.5, 6.5, 6.5])
    # Object 12 seen once: [3,3,3].
    assert np.allclose(result[12], [3.0, 3.0, 3.0])


def test_output_dtype_is_float32():
    images = _fake_images()
    masks = [[ObjectMask(mask_id=1, mask=_mask([(0, 0)]))]]

    result = _embedder_with_fake().embed_regions(images, masks)

    assert result[1].dtype == np.float32
