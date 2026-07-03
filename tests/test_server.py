"""Tests for the FastAPI query server, built via the build_app DI factory.

build_app takes the bundle + embedder as arguments, so we hand it the same kind
of hand-built fakes test_query.py uses — no scene.pkl, no SigLIP, no GPU. We
drive the app through FastAPI's TestClient (a real, in-process HTTP round trip),
so these tests exercise the JSON boundary too (NumPy -> native types).
"""

import subprocess
import sys
import textwrap

import numpy as np
from fastapi.testclient import TestClient

from lumen3d.cache import save_scene
from lumen3d.server import build_app, create_demo_app


class _FakeEmbedder:
    """Stand-in for SigLIPEmbedder: embed_text returns a fixed, chosen vector
    (ignores the query string), so the test controls which object ranks first.
    No model is ever loaded. Records every query it's asked for in `calls`, so a
    test can assert the startup warm-up fired."""

    def __init__(self, vector):
        self._vector = vector
        self.calls = []

    def embed_text(self, query):
        self.calls.append(query)
        return self._vector


def _fake_bundle():
    """A tiny 2-object scene. Object 1 points along [1, 0], object 2 along
    [0, 1]; their geometry arrays are distinct and lopsided so a join bug
    (returning the wrong object's points) can't hide."""
    embeddings = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }
    geometry = {
        1: (np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.uint8)),
        2: (np.array([[9, 9, 9]], dtype=np.float32),
            np.array([[99, 99, 99]], dtype=np.uint8)),
    }
    return {"embeddings": embeddings, "geometry": geometry, "scene": None}


def _client(query_vector=np.array([1.0, 0.0], dtype=np.float32)):
    app = build_app(_fake_bundle(), _FakeEmbedder(query_vector))
    return TestClient(app)


def test_health_reports_object_count():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "objects": 2}


def test_query_returns_winning_object_geometry():
    # Query vector points at object 1's direction -> object 1 must win, and the
    # response must carry OBJECT 1's points/colors (the join is right, not just
    # the id). float32 -> .tolist() gives floats; 0.0 == 0 so the compare holds.
    client = _client(np.array([1.0, 0.0], dtype=np.float32))
    resp = client.post("/query", json={"text": "anything"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    match = results[0]
    assert match["mask_id"] == 1
    assert np.isclose(match["score"], 1.0)
    assert match["points"] == [[0, 0, 0], [1, 1, 1]]
    assert match["colors"] == [[10, 10, 10], [20, 20, 20]]


def test_query_other_direction_wins_other_object():
    # Flip the injected query vector to object 2's direction -> object 2 wins.
    client = _client(np.array([0.0, 1.0], dtype=np.float32))
    resp = client.post("/query", json={"text": "anything"})
    match = resp.json()["results"][0]
    assert match["mask_id"] == 2
    assert match["points"] == [[9, 9, 9]]


def test_query_missing_text_field_is_422():
    # Pydantic validation: a body with no `text` is a clean 422 (bad request),
    # not a 500 (server crash).
    client = _client()
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_embedder_is_warmed_up_on_startup():
    # The lifespan warm-up only fires when the app's startup runs, which the
    # TestClient triggers only when used as a context manager (`with`). This is
    # exactly why the other tests here — built without `with` — never warm the
    # embedder and stay green.
    embedder = _FakeEmbedder(np.array([1.0, 0.0], dtype=np.float32))
    app = build_app(_fake_bundle(), embedder)

    assert embedder.calls == []                # not warmed just by building

    with TestClient(app) as client:            # entering fires startup
        assert embedder.calls == ["warmup"]    # warmed once, before any request
        resp = client.post("/query", json={"text": "anything"})
        assert resp.status_code == 200         # and normal queries still work
    assert embedder.calls == ["warmup", "anything"]


def test_create_demo_app_serves_the_frozen_scene_folder(tmp_path, monkeypatch):
    # create_demo_app is the hosted entrypoint (FR-10): it reads LUMEN3D_SCENE,
    # loads <dir>/bundle.pkl, and serves <dir>/scene.ply. Build a tiny frozen
    # folder on disk and point the env var at it. No `with` on the client, so
    # startup never fires -> the real SigLIPEmbedder it constructs is never asked
    # to load a model (and /health, /scene.ply don't need one anyway).
    scene_dir = tmp_path / "frozen"
    scene_dir.mkdir()
    b = _fake_bundle()
    save_scene(scene_dir / "bundle.pkl", b["embeddings"], b["geometry"], b["scene"])
    # write_bytes, not write_text: on Windows write_text translates \n -> \r\n,
    # but FileResponse serves the raw bytes, so compare bytes to bytes.
    (scene_dir / "scene.ply").write_bytes(b"ply\nstand-in point cloud\n")

    monkeypatch.setenv("LUMEN3D_SCENE", str(scene_dir))
    client = TestClient(create_demo_app())

    assert client.get("/health").json() == {"status": "ok", "objects": 2}
    resp = client.get("/scene.ply")            # served from the frozen folder
    assert resp.status_code == 200
    assert resp.content == b"ply\nstand-in point cloud\n"


def test_query_host_is_da3_sam2_free():
    # FR-10's core premise: the hosted query path must NEVER import DA3 or SAM2
    # (their multi-GB weights + torch-CUDA build are exactly what makes building
    # a GPU job and serving a light one). Prove it in a FRESH interpreter, so
    # another test's imports can't pollute sys.modules: import the server, build
    # an app around a fake bundle, then assert no depth_anything_3 / sam2 module
    # got pulled in. This is most meaningful under .venv-da3, where both ARE
    # installed -> it proves "available" doesn't mean "imported".
    probe = textwrap.dedent("""
        import sys
        import numpy as np
        from lumen3d.server import build_app

        class _Fake:
            def embed_text(self, query):
                return np.zeros(2, dtype="float32")

        build_app({"embeddings": {}, "geometry": {}, "scene": None}, _Fake())
        leaked = sorted(m for m in sys.modules
                        if m.split(".")[0] in {"depth_anything_3", "sam2"})
        assert not leaked, f"heavy build stack leaked into the query host: {leaked}"
        print("CLEAN")
    """)
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
