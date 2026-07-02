"""3D instance association (Option B) — identity from geometry, not tracking.

SAM2's automatic mask generator now discovers objects INDEPENDENTLY on every
frame (see `segmentation.SAM2Segmenter.segment`). So the same real object shows
up as a separate detection in each frame it appears in, with no shared id. This
module recovers identity in 3D: it unprojects every detection into a world-space
point cloud (DA3 puts all frames in one consistent world frame) and merges
detections whose clouds occupy the same region of space.

Why 3D instead of the old 2D video tracker:
  - It fixes the real limitation — objects revealed after frame 0 now get found
    and grouped, instead of being invisible to the whole pipeline.
  - For the widely-spaced keyframes we build from (stride 40-60), 3D position is
    a stronger "same object?" signal than a mask tracker trained on video-rate
    footage.

The output is drop-in for the rest of the pipeline: relabeled masks whose
`mask_id` is now a frame-stable INSTANCE id (so `embed_regions` groups correctly)
plus the per-instance geometry dict `query_scene`/the viewer already expect.
"""

import numpy as np

from .geometry import resize_mask, to_homogeneous, frame_world_points, valid_pixels


def _voxel_set(points, voxel_size):
    """The set of occupied integer voxel cells a point cloud touches.

    Snapping points to a coarse grid turns "do these two clouds overlap in
    space?" into a plain set-intersection question, and makes the comparison
    robust to the two views sampling slightly different surface points.
    """
    if len(points) == 0:
        return set()
    cells = np.floor(np.asarray(points) / voxel_size).astype(np.int64)
    cells = np.unique(cells, axis=0)
    return set(map(tuple, cells))


def _voxel_iou(a, b):
    """Intersection-over-union of two voxel sets (0.0 if either is empty)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def associate_masks_3d(
    recon,
    masks,
    conf_thr=0.0,
    voxel_frac=0.02,
    iou_thr=0.25,
    min_points=25,
):
    """Group per-frame detections into 3D instances and label everything.

    Args:
        recon: A `Reconstruction` carrying per-frame `depth`, `intrinsics`,
            `extrinsics`, `images`, `conf` (DA3 populates these).
        masks: `list[list[ObjectMask]]` straight from `Segmenter.segment` —
            per-frame detections with per-detection (non-stable) ids.
        conf_thr: Confidence floor for a pixel to become a 3D point (same knob
            fusion uses).
        voxel_frac: Voxel edge length as a fraction of the scene's bounding-box
            diagonal. DA3 world units are arbitrary (non-metric), so the voxel
            size is derived from scene scale instead of hard-coded in meters.
        iou_thr: Minimum voxel-IoU for a detection to be judged the same object
            as an existing instance. Lower = more merging.
        min_points: Detections with fewer valid 3D points than this are dropped
            as noise (from BOTH the masks and the geometry, so the two stay in
            lockstep).

    Returns:
        (relabeled_masks, geometry):
            relabeled_masks: `list[list[ObjectMask]]`, same per-frame layout as
                the input (minus dropped detections), with every `mask_id` set to
                its instance id.
            geometry: `{instance_id: (points (P,3) float32, colors (P,3) uint8)}`,
                each instance's points fused across all frames it appears in.
    """
    # Voxel size from scene scale, so iou_thr means the same thing regardless of
    # DA3's arbitrary world units.
    scene = recon.points
    if len(scene):
        diagonal = float(np.linalg.norm(scene.max(axis=0) - scene.min(axis=0)))
    else:
        diagonal = 1.0
    voxel_size = max(diagonal * voxel_frac, 1e-6)

    # --- 1. Unproject every detection to a world-space fragment ---------------
    # World points are computed ONCE per frame (vectorized) and then indexed by
    # each mask, so cost is O(frames), not O(frames * detections).
    detections = []   # (frame_idx, det_idx, points, colors, voxel_set)
    for i, frame_masks in enumerate(masks):
        c2w = np.linalg.inv(to_homogeneous(recon.extrinsics[i]))
        world = frame_world_points(recon.depth[i], recon.intrinsics[i], c2w)   # (H, W, 3)
        base_valid = valid_pixels(recon.depth[i], recon.conf[i], conf_thr)     # (H, W) bool
        shape = recon.depth[i].shape

        for d, obj in enumerate(frame_masks):
            keep = base_valid & resize_mask(obj.mask, shape)
            if int(keep.sum()) < min_points:
                continue                                    # too small -> noise, drop it
            points = world[keep].astype(np.float32)
            colors = recon.images[i][keep].astype(np.uint8)
            detections.append((i, d, points, colors, _voxel_set(points, voxel_size)))

    # --- 2. Greedy 3D clustering ---------------------------------------------
    # Walk the detections in frame order; each joins the existing instance it
    # overlaps most (if that overlap clears iou_thr), else it starts a new one.
    instances = []            # each: {"points": [...], "colors": [...], "voxels": set}
    det_to_instance = {}      # (frame_idx, det_idx) -> instance id
    for (i, d, points, colors, voxels) in detections:
        best_id, best_iou = -1, 0.0
        for inst_id, inst in enumerate(instances):
            score = _voxel_iou(voxels, inst["voxels"])
            if score > best_iou:
                best_iou, best_id = score, inst_id

        if best_id >= 0 and best_iou >= iou_thr:
            inst = instances[best_id]
            inst["points"].append(points)
            inst["colors"].append(colors)
            inst["voxels"] |= voxels
            det_to_instance[(i, d)] = best_id
        else:
            instances.append({"points": [points], "colors": [colors], "voxels": set(voxels)})
            det_to_instance[(i, d)] = len(instances) - 1

    # --- 3. Relabel masks with their instance ids ----------------------------
    # New ObjectMask objects (don't mutate the caller's), same per-frame layout,
    # dropped detections omitted.
    from .segmentation import ObjectMask   # local import avoids an import cycle
    relabeled = []
    for i, frame_masks in enumerate(masks):
        kept = []
        for d, obj in enumerate(frame_masks):
            inst_id = det_to_instance.get((i, d))
            if inst_id is None:
                continue                                    # was dropped by min_points
            kept.append(ObjectMask(mask_id=inst_id, mask=obj.mask))
        relabeled.append(kept)

    # --- 4. Fuse each instance's points across frames ------------------------
    geometry = {}
    for inst_id, inst in enumerate(instances):
        points = np.concatenate(inst["points"], axis=0).astype(np.float32)
        colors = np.concatenate(inst["colors"], axis=0).astype(np.uint8)
        geometry[inst_id] = (points, colors)

    return relabeled, geometry
