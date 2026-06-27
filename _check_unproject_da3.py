"""Throwaway check: does our unproject() match DA3's oracle on REAL data?

Run inside the DA3 venv (it needs the depth_anything_3 package):

    .venv-da3\\Scripts\\python.exe _check_unproject_da3.py

It runs DA3 once on a couple of real frames, then feeds the SAME prediction
to (a) DA3's own _depths_to_world_points_with_colors and (b) our unproject,
and checks the two point clouds agree.
"""

import numpy as np

from lumen3d.ingest import find_images
from lumen3d.backbone import DA3Backbone
from lumen3d.geometry import unproject

# DA3's private helpers — the oracle we're checking against.
from depth_anything_3.utils.export.glb import (
    _depths_to_world_points_with_colors,
    get_conf_thresh,
)

# 1) Grab a couple of real frames (full-res -> our Python loop is slow; 2 is plenty).
frames = find_images("frames_out")[:2]
print(f"frames: {[p.name for p in frames]}")

# 2) Run DA3 once. BASE model on CPU is enough for a wiring check.
backbone = DA3Backbone(weights="depth-anything/DA3-BASE", device="cpu")
model = backbone._load_model()
pred = model.inference(image=[str(p) for p in frames], export_dir=None)

# 3) Same confidence threshold DA3 itself would use.
conf_thr = get_conf_thresh(pred, getattr(pred, "sky_mask", None), 1.05, 40.0)

# 4) DA3's oracle.
da3_pts, da3_cols = _depths_to_world_points_with_colors(
    pred.depth, pred.intrinsics, pred.extrinsics,
    pred.processed_images, pred.conf, conf_thr,
)

# 5) Ours (this is the slow part — pure-Python loop over every pixel).
print("running our unproject (may take a minute on full-res frames)...")
mine_pts, mine_cols = unproject(
    pred.depth, pred.intrinsics, pred.extrinsics,
    pred.processed_images, pred.conf, conf_thr,
)

# 6) Compare.
print("shapes  ours:", mine_pts.shape, " da3:", da3_pts.shape)
print("points match:", np.allclose(mine_pts, da3_pts, atol=1e-4))
print("colors match:", np.array_equal(mine_cols, da3_cols))
