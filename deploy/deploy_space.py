"""Create the Lumen3D Hugging Face Space and upload the demo (FR-10).

Uploads via the Hub HTTP API (huggingface_hub), so large files (the ~200 MB
bundle.pkl / scene.ply) are handled server-side — no git clone, no git-lfs, no
.gitattributes. Idempotent: re-running updates the same Space.

Usage (from a venv that has huggingface_hub, e.g. .venv-da3):
    # 1. Get a WRITE token: https://huggingface.co/settings/tokens
    # 2. PowerShell:  $env:HF_TOKEN = "hf_..."
    # 3. python deploy/deploy_space.py <your-hf-username>/lumen3d
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit('usage: python deploy/deploy_space.py <hf-username>/<space-name>')
    repo_id = sys.argv[1]

    token = os.environ.get("HF_TOKEN")  # None -> falls back to a cached login
    api = HfApi(token=token)

    # Confirm we're authenticated before doing anything, with a clear message.
    who = api.whoami()
    print(f"authenticated as: {who.get('name', '?')}")

    print(f"creating Space {repo_id} (docker sdk, exist_ok)...")
    api.create_repo(repo_id=repo_id, repo_type="space",
                    space_sdk="docker", exist_ok=True)

    # The HF Space landing page (its README.md carries the sdk/app_port frontmatter).
    print("uploading README.md ...")
    api.upload_file(path_or_fileobj=str(REPO_ROOT / "deploy" / "space-README.md"),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="space")

    # Root build files.
    for name in ("Dockerfile", "requirements-host.txt", ".dockerignore"):
        print(f"uploading {name} ...")
        api.upload_file(path_or_fileobj=str(REPO_ROOT / name),
                        path_in_repo=name, repo_id=repo_id, repo_type="space")

    # Package source (imported via PYTHONPATH in the image).
    print("uploading src/ ...")
    api.upload_folder(folder_path=str(REPO_ROOT / "src"), path_in_repo="src",
                      repo_id=repo_id, repo_type="space")

    # Viewer front end — skip the stale local scene.ply (the Space serves the
    # frozen demo's cloud via the /scene.ply route, so this one is dead weight).
    print("uploading viewer/ (excluding scene.ply) ...")
    api.upload_folder(folder_path=str(REPO_ROOT / "viewer"), path_in_repo="viewer",
                      repo_id=repo_id, repo_type="space",
                      ignore_patterns=["scene.ply"])

    # The frozen scene (~390 MB) — bundle.pkl + scene.ply. Handled as LFS server-side.
    print("uploading demo/ (~390 MB, this is the slow part) ...")
    api.upload_folder(folder_path=str(REPO_ROOT / "demo"), path_in_repo="demo",
                      repo_id=repo_id, repo_type="space")

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDONE. The Space is building. Watch the logs, then open:\n  {url}")


if __name__ == "__main__":
    main()
