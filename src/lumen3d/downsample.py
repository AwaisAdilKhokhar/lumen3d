import numpy as np


def voxel_downsample(points, colors, voxel_size):
    """Thin a point cloud by keeping one averaged point per occupied voxel.

    Snaps every point to a `voxel_size`-edged grid, then collapses each occupied
    cell to a single point at the members' centroid (mean color). This is purely
    a VIEWER-WEIGHT reduction for the scene backdrop -- the per-instance geometry
    used for retrieval is untouched, so query results are unaffected. Uniform grid
    thinning (unlike random subsampling) keeps density even across the cloud.

    voxel_size is in the cloud's own (non-metric) units, so callers should derive
    it from the scene extent -- e.g. a fraction of the bbox diagonal -- the same
    scale-robust trick association.py uses.

    Returns (points, colors) as float32 / uint8, one row per occupied voxel.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)

    # Integer voxel coordinate per point, then a single voxel id per point via
    # unique-rows. return_inverse maps each original point to its voxel's id.
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n_voxels = inv.max() + 1

    # Per-voxel centroid = summed coords / member count (bincount groups by id).
    counts = np.bincount(inv, minlength=n_voxels)
    out_pts = np.empty((n_voxels, 3), dtype=np.float32)
    out_cols = np.empty((n_voxels, 3), dtype=np.uint8)
    for d in range(3):
        out_pts[:, d] = np.bincount(inv, weights=points[:, d], minlength=n_voxels) / counts
        mean_c = np.bincount(inv, weights=colors[:, d], minlength=n_voxels) / counts
        out_cols[:, d] = np.clip(np.round(mean_c), 0, 255).astype(np.uint8)

    return out_pts, out_cols
