import os
from contextlib import asynccontextmanager
from pathlib import Path

from .cache import load_scene
from .embedding import SigLIPEmbedder
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .query import query_scene
from fastapi.staticfiles import StaticFiles

# The viewer front end (index.html + its assets) lives at the repo root, not
# inside the package. Resolve it from THIS file so the server works regardless
# of the current working directory: server.py is <repo>/src/lumen3d/server.py,
# so parents[2] is the repo root.
VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


class Query(BaseModel):       # describes the JSON body: {"text": "..."}
    text: str


def build_app(bundle, embedder, scene_ply=None) -> FastAPI:
    """Build the query server around an ALREADY-LOADED scene bundle + embedder.

    Dependencies are *injected* (not constructed in here) for two payoffs:
      1. tests pass fakes — no scene.pkl on disk, no SigLIP, no GPU;
      2. a GPU-free "query-only" deployment can pass a frozen bundle + a
         text-only embedder, so the host never imports DA3/SAM2 (FR-10).
    The routes are defined INSIDE the factory, so they close over THIS call's
    `bundle` and `embedder` — each app carries its own ingredients.

    If `scene_ply` is given, the viewer's `scene.ply` request is served from
    that path (an arbitrary scene folder) instead of `viewer/scene.ply`. This
    is what `lumen3d view <folder>` needs: point the point cloud at the folder
    the user named, without copying files into `viewer/`. Left `None`, the
    static mount serves `viewer/scene.ply` as before (the demo-server default).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm the embedder ONCE, at startup, on the main thread — before the
        # server accepts any request. This kills two problems at once:
        #   1. cold-start latency: otherwise the FIRST /query pays the full
        #      SigLIP load (and possibly a weights download);
        #   2. the lazy-import race: /query runs in a worker thread, so loading
        #      the model there for the first time can flake (the transient
        #      "cannot import name 'AutoImageProcessor'"). Loading here, on the
        #      main thread before serving, removes the race.
        # We warm through the PUBLIC query path (embed_text) so we load exactly
        # what serving uses; the returned vector is discarded. No try/except:
        # if the model can't load, we WANT the server to refuse to start rather
        # than boot and 500 on every query.
        embedder.embed_text("warmup")
        yield

    app = FastAPI(lifespan=lifespan)

    if scene_ply is not None:
        scene_ply = Path(scene_ply)

        # Registered BEFORE the "/" static mount, so it wins for this one path
        # (Starlette matches routes in order; the mount would otherwise catch it).
        @app.get("/scene.ply")
        def scene_point_cloud():
            return FileResponse(scene_ply)

    @app.get("/health")                   # "when someone GETs /health, run this"
    def health():
        return {"status": "ok", "objects": len(bundle["embeddings"])}

    @app.post("/query")
    def query(q: Query):          # FastAPI parses the body into `q`
        user_text = q.text
        result = query_scene(user_text, bundle, embedder, top_k=1)
        result_dict = {}
        result_dict["results"] = []
        for mask_id, score, points, colors in result:
            result_dict["results"].append(
                {"mask_id": int(mask_id), "score": float(score),
                 "points": points.tolist(), "colors": colors.tolist()}
            )
        return result_dict

    # Mount the viewer LAST, after the API routes, so "/" doesn't shadow them.
    # check_dir=False keeps build_app callable from any working directory
    # (e.g. under pytest, where the cwd may not be the project root).
    app.mount("/", StaticFiles(directory=str(VIEWER_DIR), html=True,
                               check_dir=False), name="viewer")
    return app


def create_default_app() -> FastAPI:
    """Real wiring: load the pickled scene + the SigLIP embedder, then build.

    Kept separate from build_app so that *importing* this module touches neither
    the disk nor a model — only running the server does. Run with:
        uvicorn lumen3d.server:create_default_app --factory --port 8000
    """
    return build_app(load_scene("scene.pkl"), SigLIPEmbedder())


def create_demo_app() -> FastAPI:
    """Query-only entrypoint for the hosted, GPU-free demo (FR-10).

    Points at a FROZEN scene FOLDER — `<dir>/bundle.pkl` (the per-instance
    embedding table + geometry a query needs) and `<dir>/scene.ply` (the
    backdrop the viewer renders). It loads only the LIGHT "consume" stack: the
    pickled bundle + SigLIP's text tower. Importing this module or building this
    app touches NEITHER DA3 NOR SAM2 — that heavy "produce" stack is why building
    is an offline CLI/Colab job; serving a pre-built scene is not. (Proven, not
    assumed: see test_query_host_is_da3_sam2_free.)

    Differs from create_default_app in two ways that matter for the frozen demo:
    it reads a scene *folder* (not the legacy cwd-relative `scene.pkl`), and it
    serves that folder's `scene.ply` so the viewer renders the demo's own cloud.

    The folder comes from the LUMEN3D_SCENE env var (default "demo"), so the
    Docker/Space host can point it wherever the frozen scene was copied without a
    code change. Run with:
        uvicorn lumen3d.server:create_demo_app --factory --host 0.0.0.0 --port 7860
    """
    scene_dir = Path(os.environ.get("LUMEN3D_SCENE", "demo"))
    bundle = load_scene(scene_dir / "bundle.pkl")
    return build_app(bundle, SigLIPEmbedder(), scene_ply=scene_dir / "scene.ply")
