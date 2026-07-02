"""Produce side: turn frames into a queryable scene bundle.

`build_scene` is the partner to `query.query_scene`. It runs the four heavy
machines (DA3 -> SAM2 -> fusion -> SigLIP) on a set of frames and assembles
their outputs into the same bundle dict that `cache.load_scene` reads back and
`query_scene` searches.
"""

from .association import associate_masks_3d
from .cache import save_scene


def build_scene(
    frames,
    backbone,
    segmenter,
    embedder,
    conf_thr=0.0,
    voxel_frac=0.01,
    iou_thr=0.25,
    sim_thr=0.85,
    min_points=25,
    save_path=None,
) -> dict:
    """Build a scene bundle from frames by running the full produce pipeline.

    Pure orchestration: it owns no model math, it just calls the machines in
    order and packs the results. The three models are passed in (not built
    here) so tests can hand in fakes and so backbones stay swappable (FR-9).

    Args:
        frames: List of frame paths (from `ingest.load_frames`).
        backbone: A `Backbone` (e.g. `DA3Backbone`) — gives 3D geometry.
        segmenter: A `Segmenter` (e.g. `SAM2Segmenter`) — gives per-frame masks.
        embedder: An `Embedder` (e.g. `SigLIPEmbedder`) — gives meaning vectors.
        conf_thr: Confidence threshold passed down to unprojection.
        voxel_frac, iou_thr, sim_thr, min_points: 3D-association knobs — see
            `association.associate_masks_3d` (voxel size as a fraction of scene
            diagonal, geometric + semantic merge thresholds, and the minimum
            points to keep a detection).
        save_path: If given, also pickle the bundle to this path via
            `save_scene`. If None, nothing is written to disk.

    Returns:
        A bundle dict with three compartments:
            "embeddings": {instance_id: (768,) float32}   — meaning per object
            "geometry":   {instance_id: (points, colors)} — 3D points per object
            "scene":      (points, colors)                — the full backdrop cloud
    """
    recon = backbone.reconstruct(frames)          # DA3  -> Reconstruction
    masks = segmenter.segment(frames)             # SAM2 -> per-frame detections

    # One embedding per detection: segment() gives every detection a unique id,
    # so embed_regions (which groups by id) returns exactly one vector each.
    # Association needs these up front to gate merges on meaning, not just space.
    det_embeddings = embedder.embed_regions(recon.images, masks)

    # Group detections into 3D instances by geometry AND meaning: relabels masks
    # with frame-stable instance ids, fuses each instance's points, and averages
    # its member embeddings — all keyed the same way, in one pass.
    masks, geometry, embeddings = associate_masks_3d(
        recon, masks, det_embeddings, conf_thr=conf_thr,
        voxel_frac=voxel_frac, iou_thr=iou_thr, sim_thr=sim_thr, min_points=min_points,
    )
    scene = (recon.points, recon.colors)          # the whole room, as a backdrop

    bundle = {
        "embeddings": embeddings,
        "geometry": geometry,
        "scene": scene,
    }

    if save_path is not None:
        save_scene(save_path, embeddings, geometry, scene)

    return bundle
