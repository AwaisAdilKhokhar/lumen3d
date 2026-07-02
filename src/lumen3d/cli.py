"""Command-line front door for Lumen3D.

Turns the library functions into a real `lumen3d` command with subcommands:

    lumen3d reconstruct <video-or-folder> -o scene/   # build a queryable scene
    lumen3d query scene/ "the red backpack"           # (later)
    lumen3d view scene/                               # (later)

The CLI owns no pipeline math — it only parses arguments and calls the front
doors (`load_frames`, `build_scene`, `export_ply`). The heavy model imports live
*inside* the `reconstruct` branch so the other commands don't need DA3/SAM2
installed just to run.
"""

import argparse
from pathlib import Path

from .ingest import load_frames, downscale_frames


def main():
    parser = argparse.ArgumentParser(prog="lumen3d")
    subparsers = parser.add_subparsers(dest="command")

    # the "reconstruct" window: input frames -> a scene folder on disk
    recon = subparsers.add_parser(
        "reconstruct",
        help="Build a queryable scene from a video, an image folder, or a single image.",
    )
    recon.add_argument(
        "input",
        help="Path to a video file, a folder of images, or a single image. "
             "(A single image gives a monocular / 2.5D reconstruction.)",
    )
    recon.add_argument(
        "-o", "--output", default="scene",
        help="Folder to write the scene into (default: 'scene').",
    )
    recon.add_argument(
        "--stride", type=int, default=10,
        help="For a video: keep every Nth frame (default: 10). Ignored for folders.",
    )
    recon.add_argument(
        "--max-width", type=int, default=1024,
        help="Downscale frames to at most this width before building, to fit GPU "
             "memory (default: 1024). Use 0 to disable downscaling.",
    )
    recon.add_argument(
        "--model", default="depth-anything/DA3-LARGE",
        help="DA3 backbone weights (default: depth-anything/DA3-LARGE).",
    )
    recon.add_argument(
        "--conf", type=float, default=0.0,
        help="Confidence threshold passed to fusion's unprojection (default: 0.0).",
    )

    # the "query" window: a built scene folder + text -> ranked matches printed
    q = subparsers.add_parser(
        "query",
        help="Search a built scene by text and print the ranked matches.",
    )
    q.add_argument("scene", help="Path to a scene folder (containing bundle.pkl).")
    q.add_argument("text", help='The text query, e.g. "the red backpack".')
    q.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="How many matches to print, best first (default: 5).",
    )

    # the "view" window: launch the web viewer + query server on a scene folder
    view = subparsers.add_parser(
        "view",
        help="Launch the web viewer and query server on a scene folder.",
    )
    view.add_argument(
        "scene",
        help="Path to a scene folder (containing bundle.pkl and scene.ply).",
    )
    view.add_argument(
        "--host", default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    view.add_argument(
        "--port", type=int, default=8000,
        help="Port to serve on (default: 8000).",
    )

    args = parser.parse_args()

    if args.command == "reconstruct":
        run_reconstruct(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "view":
        run_view(args)
    else:
        # No command (or an unknown one) -> show help instead of doing nothing.
        parser.print_help()


def run_reconstruct(args):
    """Build a scene from `args.input` and write it into `args.output/`.

    Writes two files into the output folder:
        bundle.pkl  -- the scene bundle (embeddings + geometry + backdrop),
                       what `query` will load.
        scene.ply   -- the point cloud, what the web viewer draws.
    """
    # Heavy models: imported here (not at module top) so `query`/`view` can run
    # in the light venv without DA3/SAM2 installed. This branch needs .venv-da3.
    from .backbone import DA3Backbone
    from .segmentation import SAM2Segmenter
    from .embedding import SigLIPEmbedder
    from .build import build_scene
    from .export import export_ply

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "bundle.pkl"
    ply_path = out_dir / "scene.ply"

    print(f"[1/3] loading frames from {args.input!r} ...")
    frames = load_frames(args.input, stride=args.stride)
    if args.max_width and args.max_width > 0:
        frames = downscale_frames(frames, max_width=args.max_width)
    print(f"      {len(frames)} frames (max width {args.max_width}px).")

    print("[2/3] building scene (DA3 -> SAM2 -> 3D association -> SigLIP) ...")
    bundle = build_scene(
        frames,
        DA3Backbone(args.model),
        SAM2Segmenter(),
        SigLIPEmbedder(),
        conf_thr=args.conf,
        save_path=bundle_path,
    )

    print(f"[3/3] writing point cloud -> {ply_path} ...")
    export_ply(ply_path, *bundle["scene"])

    n_objects = len(bundle["geometry"])
    n_points = len(bundle["scene"][0])
    print(f"done. scene -> {out_dir}/  ({n_objects} objects, {n_points} points)")


def run_query(args):
    """Load `args.scene/bundle.pkl`, run one text query, print ranked matches.

    Unlike `reconstruct`, this touches no DA3/SAM2 — only SigLIP's *text* tower
    (to turn the phrase into a vector) and pure-NumPy cosine. So the heavy
    backbone imports stay out of this branch. (It still needs torch for SigLIP,
    so today it runs in .venv-da3; a torch-light query path is future FR-10 work.)
    """
    from .cache import load_scene
    from .embedding import SigLIPEmbedder
    from .query import query_scene

    bundle_path = Path(args.scene) / "bundle.pkl"
    if not bundle_path.exists():
        raise SystemExit(
            f"No bundle.pkl in {args.scene!r} — run `lumen3d reconstruct` first."
        )

    bundle = load_scene(bundle_path)
    results = query_scene(args.text, bundle, SigLIPEmbedder(), top_k=args.top_k)

    if not results:
        print("This scene has no objects to match.")
        return

    print(f'Top {len(results)} match(es) for {args.text!r}:')
    for rank, (mask_id, score, points, colors) in enumerate(results, start=1):
        print(f"  {rank}. object {mask_id:>3}   score {score:.4f}   ({len(points)} points)")


def run_view(args):
    """Serve the web viewer + query API for the scene folder `args.scene`.

    Loads the folder's bundle.pkl, wires it into the FastAPI app (with the
    folder's scene.ply as the point cloud the browser draws), and runs uvicorn.
    Runs in .venv-da3 (needs SigLIP's text tower for /query, and uvicorn).
    """
    from .cache import load_scene
    from .embedding import SigLIPEmbedder
    from .server import build_app
    import uvicorn

    scene_dir = Path(args.scene)
    bundle_path = scene_dir / "bundle.pkl"
    ply_path = scene_dir / "scene.ply"
    if not bundle_path.exists():
        raise SystemExit(
            f"No bundle.pkl in {args.scene!r} — run `lumen3d reconstruct` first."
        )
    if not ply_path.exists():
        raise SystemExit(f"No scene.ply in {args.scene!r} — the viewer needs it.")

    bundle = load_scene(bundle_path)
    app = build_app(bundle, SigLIPEmbedder(), scene_ply=ply_path)

    print(f"serving {scene_dir}/ at http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
