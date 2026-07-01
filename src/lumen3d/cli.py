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

from .ingest import load_frames


def main():
    parser = argparse.ArgumentParser(prog="lumen3d")
    subparsers = parser.add_subparsers(dest="command")

    # the "reconstruct" window: input frames -> a scene folder on disk
    recon = subparsers.add_parser(
        "reconstruct",
        help="Build a queryable scene from a video or image folder.",
    )
    recon.add_argument("input", help="Path to a video file or a folder of images.")
    recon.add_argument(
        "-o", "--output", default="scene",
        help="Folder to write the scene into (default: 'scene').",
    )

    args = parser.parse_args()

    if args.command == "reconstruct":
        run_reconstruct(args)
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
    frames = load_frames(args.input)
    print(f"      {len(frames)} frames.")

    print("[2/3] building scene (DA3 -> SAM2 -> fusion -> SigLIP) ...")
    bundle = build_scene(
        frames,
        DA3Backbone(),
        SAM2Segmenter(),
        SigLIPEmbedder(),
        save_path=bundle_path,
    )

    print(f"[3/3] writing point cloud -> {ply_path} ...")
    export_ply(ply_path, *bundle["scene"])

    n_objects = len(bundle["geometry"])
    n_points = len(bundle["scene"][0])
    print(f"done. scene -> {out_dir}/  ({n_objects} objects, {n_points} points)")


if __name__ == "__main__":
    main()
