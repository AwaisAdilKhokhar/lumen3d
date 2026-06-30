"""Tests for save_scene / load_scene (the disk cache for a finished scene).

Pure-logic tests: no DA3, no SAM2, no GPU. We hand-build a tiny fake scene
(the two object-keyed dicts + an optional full cloud), write it to a temp file,
read it back, and check it survived the round trip *identically* — same keys,
same arrays, same dtypes. pytest's `tmp_path` fixture gives us a throwaway
directory so we never touch real files.
"""

import numpy as np

from lumen3d.cache import save_scene, load_scene


def _fake_scene():
    """A minimal scene: two objects (ids 7 and 12), each with an embedding and a
    little ragged point cloud, plus a small full-scene backdrop."""
    embeddings = {
        7:  np.arange(768, dtype=np.float32),
        12: np.full(768, 0.5, dtype=np.float32),
    }
    geometry = {
        # object 7: 3 points; object 12: 1 point -> deliberately ragged.
        7:  (np.array([[0, 2, 2], [0, 2, 2], [2, 2, 2]], dtype=np.float32),
             np.array([[30, 30, 30], [70, 70, 70], [80, 80, 80]], dtype=np.uint8)),
        12: (np.array([[0, 0, 2]], dtype=np.float32),
             np.array([[10, 10, 10]], dtype=np.uint8)),
    }
    scene = (
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    return embeddings, geometry, scene


def _assert_geometry_equal(a, b):
    """Two geometry dicts match: same ids, and each (points, colors) pair is
    element-for-element equal with the same dtypes."""
    assert a.keys() == b.keys()
    for mask_id in a:
        pa, ca = a[mask_id]
        pb, cb = b[mask_id]
        assert np.array_equal(pa, pb)
        assert np.array_equal(ca, cb)
        assert pa.dtype == pb.dtype
        assert ca.dtype == cb.dtype


def test_round_trip_preserves_everything(tmp_path):
    # Arrange
    embeddings, geometry, scene = _fake_scene()
    path = tmp_path / "scene.pkl"

    # Act: save, then load back into a fresh bundle.
    save_scene(path, embeddings, geometry, scene)
    bundle = load_scene(path)

    # Assert: the three pieces came back under the expected keys.
    assert set(bundle.keys()) == {"embeddings", "geometry", "scene"}

    # embeddings: same ids, same vectors, same dtype.
    assert bundle["embeddings"].keys() == embeddings.keys()
    for mask_id in embeddings:
        assert np.array_equal(bundle["embeddings"][mask_id], embeddings[mask_id])
        assert bundle["embeddings"][mask_id].dtype == embeddings[mask_id].dtype

    # geometry: ragged clouds survive intact, row-aligned, right dtypes.
    _assert_geometry_equal(bundle["geometry"], geometry)

    # scene backdrop: both arrays round-trip.
    pts, cols = bundle["scene"]
    assert np.array_equal(pts, scene[0])
    assert np.array_equal(cols, scene[1])


def test_scene_defaults_to_none_when_omitted(tmp_path):
    # The full-cloud backdrop is optional; omitting it stores None, not a crash.
    embeddings, geometry, _ = _fake_scene()
    path = tmp_path / "no_backdrop.pkl"

    save_scene(path, embeddings, geometry)  # no `scene` argument
    bundle = load_scene(path)

    assert bundle["scene"] is None
    # the other two are still fully present.
    assert bundle["embeddings"].keys() == embeddings.keys()
    _assert_geometry_equal(bundle["geometry"], geometry)


def test_accepts_string_path(tmp_path):
    # `path` may be a str or a Path; open() handles both. Prove the str case.
    embeddings, geometry, scene = _fake_scene()
    path = str(tmp_path / "as_string.pkl")

    save_scene(path, embeddings, geometry, scene)
    bundle = load_scene(path)

    _assert_geometry_equal(bundle["geometry"], geometry)
