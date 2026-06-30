from .cache import load_scene
from .embedding import SigLIPEmbedder
from fastapi import FastAPI
from pydantic import BaseModel
from .query import query_scene
from fastapi.staticfiles import StaticFiles


class Query(BaseModel):       # describes the JSON body: {"text": "..."}
    text: str


def build_app(bundle, embedder) -> FastAPI:
    """Build the query server around an ALREADY-LOADED scene bundle + embedder.

    Dependencies are *injected* (not constructed in here) for two payoffs:
      1. tests pass fakes — no scene.pkl on disk, no SigLIP, no GPU;
      2. a GPU-free "query-only" deployment can pass a frozen bundle + a
         text-only embedder, so the host never imports DA3/SAM2 (FR-10).
    The routes are defined INSIDE the factory, so they close over THIS call's
    `bundle` and `embedder` — each app carries its own ingredients.
    """
    app = FastAPI()

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
    app.mount("/", StaticFiles(directory="viewer", html=True, check_dir=False),
              name="viewer")
    return app


def create_default_app() -> FastAPI:
    """Real wiring: load the pickled scene + the SigLIP embedder, then build.

    Kept separate from build_app so that *importing* this module touches neither
    the disk nor a model — only running the server does. Run with:
        uvicorn lumen3d.server:create_default_app --factory --port 8000
    """
    return build_app(load_scene("scene.pkl"), SigLIPEmbedder())
