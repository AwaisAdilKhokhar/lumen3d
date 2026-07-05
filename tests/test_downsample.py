"""Tests for voxel_downsample (thin the viewer cloud, one point per voxel).

Pure NumPy, no GPU. We hand-place points into known voxels so we already know
how many should survive and where their centroids land.
"""

import numpy as np

from lumen3d.downsample import voxel_downsample


def test_collapses_each_voxel_to_its_centroid():
    # Two clusters, far apart -> with voxel_size 1.0 they fall in different
    # voxels and must collapse to two points at their respective centroids.
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],   # same voxel as the first -> centroid (0.15,...)
        [5.0, 5.0, 5.0],   # a second, distant voxel
    ], dtype=np.float32)
    colors = np.array([
        [100, 100, 100],
        [200, 200, 200],   # mean with the first -> 150
        [10, 20, 30],
    ], dtype=np.uint8)

    out_pts, out_cols = voxel_downsample(points, colors, voxel_size=1.0)

    assert len(out_pts) == 2
    assert out_pts.dtype == np.float32 and out_cols.dtype == np.uint8
    # unique() sorts voxel keys, so the near-origin cluster comes first.
    order = np.lexsort((out_pts[:, 2], out_pts[:, 1], out_pts[:, 0]))
    a, b = out_pts[order], out_cols[order]
    assert np.allclose(a[0], [0.15, 0.15, 0.15])
    assert np.array_equal(b[0], [150, 150, 150])
    assert np.allclose(a[1], [5.0, 5.0, 5.0])


def test_never_grows_and_a_coarser_grid_thins_more():
    rng = np.random.default_rng(0)
    points = rng.uniform(-10, 10, size=(5000, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(5000, 3)).astype(np.uint8)

    fine = voxel_downsample(points, colors, voxel_size=0.5)[0]
    coarse = voxel_downsample(points, colors, voxel_size=2.0)[0]

    assert len(fine) <= len(points)
    assert len(coarse) <= len(fine)          # bigger voxels merge more points
    # survivors must stay inside the original bounding box (centroids, not new points)
    assert (coarse.min(axis=0) >= points.min(axis=0) - 1e-4).all()
    assert (coarse.max(axis=0) <= points.max(axis=0) + 1e-4).all()
