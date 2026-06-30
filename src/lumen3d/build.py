"""Produce side: turn frames into a queryable scene bundle.

`build_scene` is the partner to `query.query_scene`. It runs the four heavy
machines (DA3 -> SAM2 -> fusion -> SigLIP) on a set of frames and assembles
their outputs into the same bundle dict that `cache.load_scene` reads back and
`query_scene` searches.
"""

from .fusion import fuse_masks_to_3d
from .cache import save_scene


def build_scene(frames, backbone, segmenter, embedder, conf_thr=0.0, save_path=None) -> dict:
    """Build a scene bundle from frames by running the full produce pipeline.

    Pure orchestration: it owns no model math, it just calls the machines in
    order and packs the results. The three models are passed in (not built
    here) so tests can hand in fakes and so backbones stay swappable (FR-9).

    Args:
        frames: List of frame paths (from `ingest.load_frames`).
        backbone: A `Backbone` (e.g. `DA3Backbone`) — gives 3D geometry.
        segmenter: A `Segmenter` (e.g. `SAM2Segmenter`) — gives per-frame masks.
        embedder: An `Embedder` (e.g. `SigLIPEmbedder`) — gives meaning vectors.
        conf_thr: Confidence threshold passed down to fusion's unprojection.
        save_path: If given, also pickle the bundle to this path via
            `save_scene`. If None, nothing is written to disk.

    Returns:
        A bundle dict with three compartments:
            "embeddings": {mask_id: (768,) float32}   — meaning per object
            "geometry":   {mask_id: (points, colors)} — 3D points per object
            "scene":      (points, colors)            — the full backdrop cloud
    """
    recon = backbone.reconstruct(frames)          # DA3  -> Reconstruction
    masks = segmenter.segment(frames)             # SAM2 -> list[list[ObjectMask]]

    geometry = fuse_masks_to_3d(recon, masks, conf_thr)     # mask + depth -> 3D points
    embeddings = embedder.embed_regions(recon.images, masks)  # crops -> meaning vectors
    scene = (recon.points, recon.colors)          # the whole room, as a backdrop

    bundle = {
        "embeddings": embeddings,
        "geometry": geometry,
        "scene": scene,
    }

    if save_path is not None:
        save_scene(save_path, embeddings, geometry, scene)

    return bundle
