"""Tests for similarity (cosine-rank objects against a text query vector).

Pure-logic tests: no SigLIP, no GPU. We hand-build a query vector and a tiny
dict of object embeddings whose directions we *chose*, so we already know the
correct ranking. The lopsided case is object D: same direction as the query but
twice as long — it must still score 1.0, which only holds if the norm division
is really there (a missing-normalization bug would let length leak in).
"""

import numpy as np

from lumen3d.query import similarity


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
