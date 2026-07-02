"""3D instance association (Option B) — identity from geometry AND meaning.

SAM2's automatic mask generator now discovers objects INDEPENDENTLY on every
frame (see `segmentation.SAM2Segmenter.segment`). So the same real object shows
up as a separate detection in each frame it appears in, with no shared id. This
module recovers identity: it unprojects every detection into a world-space point
cloud (DA3 puts all frames in one consistent world frame) and merges detections
that occupy the same region of space AND look alike.

Two gates, both required to merge (this is the ConceptGraphs recipe):
  - Geometric: the detections' voxel sets overlap enough (voxel-IoU >= iou_thr).
    Necessary — two chairs across the room shouldn't merge just because they
    look identical.
  - Semantic: their SigLIP embeddings are similar enough (cosine >= sim_thr).
    Necessary — a door and a trash can standing next to each other overlap in
    space (coarse voxels bridge them) but look nothing alike, so meaning keeps
    them apart. Geometry alone used to mis-merge exactly this case.

Why 3D instead of the old 2D video tracker:
  - It fixes the real limitation — objects revealed after frame 0 now get found
    and grouped, instead of being invisible to the whole pipeline.
  - For the widely-spaced keyframes we build from (stride 40-60), 3D position is
    a stronger "same object?" signal than a mask tracker trained on video-rate
    footage.

The output is drop-in for the rest of the pipeline: relabeled masks whose
`mask_id` is now a frame-stable INSTANCE id, the per-instance geometry dict, and
the per-instance embedding dict — all keyed the same way, ready for the bundle.
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


def _cosine(a, b):
    """Cosine similarity of two vectors (0.0 if either has zero norm)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    det_embeddings,
    conf_thr=0.0,
    voxel_frac=0.01,
    iou_thr=0.25,
    sim_thr=0.85,
    min_points=25,
):
    """Group per-frame detections into 3D instances and label everything.

    A detection joins an existing instance only if it clears BOTH gates against
    that instance: geometric overlap (voxel-IoU >= iou_thr) and embedding
    similarity (cosine >= sim_thr). Requiring both is what keeps a door and an
    adjacent trash can — which overlap in space but look nothing alike — as
    separate objects.

    Args:
        recon: A `Reconstruction` carrying per-frame `depth`, `intrinsics`,
            `extrinsics`, `images`, `conf` (DA3 populates these).
        masks: `list[list[ObjectMask]]` straight from `Segmenter.segment` —
            per-frame detections with per-detection (non-stable) ids.
        det_embeddings: `{detection_id: (D,) vector}` — one SigLIP embedding per
            detection, keyed by the raw `mask_id` from `segment`. Because
            `segment` gives every detection a unique id, `embed_regions` on the
            raw masks produces exactly this. A detection missing from here (e.g.
            an empty crop) is dropped, so all outputs stay in lockstep.
        conf_thr: Confidence floor for a pixel to become a 3D point.
        voxel_frac: Voxel edge length as a fraction of the scene's bounding-box
            diagonal. DA3 world units are arbitrary (non-metric), so the voxel
            size is derived from scene scale instead of hard-coded in meters.
            This is the main lever for spatially-adjacent objects: too coarse and
            a door and the trash can in front of it fall in the same voxels and
            merge regardless of the semantic gate; finer keeps them apart.
        iou_thr: Minimum voxel-IoU to consider two detections the same object.
        sim_thr: Minimum cosine similarity (on SigLIP embeddings) to consider two
            detections the same object. Raise it to split more aggressively.
        min_points: Detections with fewer valid 3D points than this are dropped
            as noise.

    Returns:
        (relabeled_masks, geometry, embeddings), all keyed by the same instance
        ids:
            relabeled_masks: `list[list[ObjectMask]]`, same per-frame layout as
                the input (minus dropped detections), every `mask_id` set to its
                instance id.
            geometry: `{instance_id: (points (P,3) float32, colors (P,3) uint8)}`,
                each instance's points fused across all frames it appears in.
            embeddings: `{instance_id: (D,) float32}`, the mean of the instance's
                member detection embeddings.
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
    # each mask, so cost is O(frames), not O(frames * detections). We also carry
    # each detection's embedding along for the semantic gate.
    detections = []   # (frame_idx, det_idx, points, colors, voxel_set, embedding)
    for i, frame_masks in enumerate(masks):
        c2w = np.linalg.inv(to_homogeneous(recon.extrinsics[i]))
        world = frame_world_points(recon.depth[i], recon.intrinsics[i], c2w)   # (H, W, 3)
        base_valid = valid_pixels(recon.depth[i], recon.conf[i], conf_thr)     # (H, W) bool
        shape = recon.depth[i].shape

        for d, obj in enumerate(frame_masks):
            emb = det_embeddings.get(obj.mask_id)
            if emb is None:
                continue                                    # no embedding -> can't gate, drop it
            keep = base_valid & resize_mask(obj.mask, shape)
            if int(keep.sum()) < min_points:
                continue                                    # too small -> noise, drop it
            points = world[keep].astype(np.float32)
            colors = recon.images[i][keep].astype(np.uint8)
            emb = np.asarray(emb, dtype=np.float32)
            detections.append((i, d, points, colors, _voxel_set(points, voxel_size), emb))

    # --- 2. Greedy clustering on geometry AND meaning ------------------------
    # Walk detections in frame order; each joins the existing instance it
    # overlaps most, but only among instances it ALSO matches semantically.
    # Each instance keeps a running embedding mean (sum / count) to compare against.
    instances = []            # {"points":[], "colors":[], "voxels":set, "emb_sum":vec, "emb_count":int}
    det_to_instance = {}      # (frame_idx, det_idx) -> instance id
    for (i, d, points, colors, voxels, emb) in detections:
        best_id, best_iou = -1, 0.0
        for inst_id, inst in enumerate(instances):
            iou = _voxel_iou(voxels, inst["voxels"])
            if iou < iou_thr:
                continue                                    # geometry gate
            mean_emb = inst["emb_sum"] / inst["emb_count"]
            if _cosine(emb, mean_emb) < sim_thr:
                continue                                    # semantic gate
            if iou > best_iou:                              # both gates passed; prefer most overlap
                best_iou, best_id = iou, inst_id

        if best_id >= 0:
            inst = instances[best_id]
            inst["points"].append(points)
            inst["colors"].append(colors)
            inst["voxels"] |= voxels
            inst["emb_sum"] += emb
            inst["emb_count"] += 1
            det_to_instance[(i, d)] = best_id
        else:
            instances.append({
                "points": [points], "colors": [colors], "voxels": set(voxels),
                "emb_sum": emb.copy(), "emb_count": 1,
            })
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
                continue                                    # was dropped (no embedding / too small)
            kept.append(ObjectMask(mask_id=inst_id, mask=obj.mask))
        relabeled.append(kept)

    # --- 4. Fuse each instance's points and average its embedding ------------
    geometry = {}
    embeddings = {}
    for inst_id, inst in enumerate(instances):
        points = np.concatenate(inst["points"], axis=0).astype(np.float32)
        colors = np.concatenate(inst["colors"], axis=0).astype(np.uint8)
        geometry[inst_id] = (points, colors)
        embeddings[inst_id] = (inst["emb_sum"] / inst["emb_count"]).astype(np.float32)

    return relabeled, geometry, embeddings
