"""Tests for similarity (cosine-rank objects against a text query vector).

Pure-logic tests: no SigLIP, no GPU. We hand-build a query vector and a tiny
dict of object embeddings whose directions we *chose*, so we already know the
correct ranking. The lopsided case is object D: same direction as the query but
twice as long — it must still score 1.0, which only holds if the norm division
is really there (a missing-normalization bug would let length leak in).
"""

import numpy as np

from lumen3d.query import similarity, query_scene


def _fake_embeddings():
    """Four objects positioned by hand around the query direction [1, 0, 0]:
    A identical, B perpendicular, C opposite, D same-direction-but-longer."""
    return {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # A: cosine  1.0
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # B: cosine  0.0
        3: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # C: cosine -1.0
        4: np.array([2.0, 0.0, 0.0], dtype=np.float32),   # D: cosine  1.0 (longer)
    }


def test_ranks_by_cosine_highest_first():
    # Arrange
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = _fake_embeddings()

    # Act
    ranked = similarity(query, embeddings)

    # Assert: A and D tie at the top (1.0), then B (0.0), then C (-1.0) last.
    ids = [mask_id for mask_id, _ in ranked]
    assert ids[:2] == [1, 4] or ids[:2] == [4, 1]   # the two 1.0s, either order
    assert ids[2] == 2
    assert ids[3] == 3


def test_normalization_makes_length_irrelevant():
    # The killer case: D points the same way as the query but is twice as long.
    # If the norms were dropped, D's raw dot (2.0) would beat A's (1.0); with
    # proper cosine both are exactly 1.0.
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = _fake_embeddings()

    scores = dict(similarity(query, embeddings))

    assert np.isclose(scores[1], 1.0)
    assert np.isclose(scores[4], 1.0)
    assert np.isclose(scores[2], 0.0)
    assert np.isclose(scores[3], -1.0)


def test_scores_are_floats_and_ids_are_ints():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = _fake_embeddings()

    ranked = similarity(query, embeddings)

    for mask_id, score in ranked:
        assert isinstance(mask_id, int)
        assert isinstance(score, float)   # plain Python float, not np.float32


def test_empty_embeddings_returns_empty_list():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert similarity(query, {}) == []


# --- query_scene: the front door (embed_text -> rank -> join to geometry) ---
#
# query_scene never touches the GPU: it only calls embedder.embed_text and does
# dict lookups. So we fake the embedder with a stand-in whose embed_text returns
# a vector WE chose (same trick as faking _embed_image in test_embedding.py),
# and hand-build a 2-object scene whose embedding directions we picked. With the
# fake query pointing straight at object 1, we already know object 1 must win.


class _FakeEmbedder:
    """Stand-in for SigLIPEmbedder: embed_text returns a fixed, chosen vector
    (ignores the query string) so the test controls which object ranks first.
    No model is ever loaded."""

    def __init__(self, vector):
        self._vector = vector

    def embed_text(self, query):
        return self._vector


def _fake_bundle():
    """A tiny 2-object scene. Object 1 points along [1, 0], object 2 along
    [0, 1]; their geometry arrays are distinct and lopsided so a join bug
    (returning the wrong object's points) can't hide."""
    embeddings = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }
    geometry = {
        1: (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.uint8)),
        2: (np.array([[9, 9, 9]], dtype=np.float32),
            np.array([[99, 99, 99]], dtype=np.uint8)),
    }
    return {"embeddings": embeddings, "geometry": geometry, "scene": None}


def test_query_scene_returns_winner_with_its_geometry():
    # Arrange: query points at object 1's direction.
    bundle = _fake_bundle()
    embedder = _FakeEmbedder(np.array([1.0, 0.0], dtype=np.float32))

    # Act: default top_k=1.
    results = query_scene("anything", bundle, embedder)

    # Assert: one result, it's object 1, and the points/colors are object 1's
    # (the join is correct, not just the id).
    assert len(results) == 1
    mask_id, score, points, colors = results[0]
    assert mask_id == 1
    assert np.isclose(score, 1.0)
    assert np.array_equal(points, bundle["geometry"][1][0])
    assert np.array_equal(colors, bundle["geometry"][1][1])


def test_query_scene_top_k_returns_ranked_list():
    bundle = _fake_bundle()
    embedder = _FakeEmbedder(np.array([1.0, 0.0], dtype=np.float32))

    results = query_scene("anything", bundle, embedder, top_k=2)

    # Two results, best first: object 1 (cosine 1.0) then object 2 (cosine 0.0).
    ids = [mask_id for mask_id, _, _, _ in results]
    assert ids == [1, 2]


def test_query_scene_top_k_larger_than_scene_does_not_crash():
    # The slicing-safety case: asking for more matches than objects exist must
    # return all of them, not raise IndexError.
    bundle = _fake_bundle()
    embedder = _FakeEmbedder(np.array([1.0, 0.0], dtype=np.float32))

    results = query_scene("anything", bundle, embedder, top_k=5)

    assert len(results) == 2
