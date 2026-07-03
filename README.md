# Lumen3D

> Turn a phone video into a language-queryable 3D scene.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Space-yellow)](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Lumen3D takes a casual video walkthrough (or a set of photos) and produces an
interactive, language-queryable 3D scene. Type a phrase — *"the red backpack"*,
*"a wooden chair"* — and the matching object lights up in 3D in your browser.

**Status:** 🚧 Early development (v0.1). Built in public as a learning + portfolio project.

---

## 🔦 Live demo

**[Try it in your browser →](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)** — no
install, no GPU, no build step. Explore a real reconstructed room and search it by typing a phrase.

<!-- TEASER GIF (TODO): record ~5 s of typing a query and the object lighting up in the
     viewer, save as docs/teaser.gif, and embed here:  ![Lumen3D demo](docs/teaser.gif) -->

> Runs on a free CPU Space: it sleeps after inactivity (first load may take ~30 s to wake),
> and the showcase scene is a large point cloud, so give it a moment to render.

---

## How it works

```
Video / photos
  → pick good frames                 (keyframe sampling)
  → reconstruct a 3D point cloud     (Depth Anything 3)
  → outline objects per frame        (SAM 2)
  → embed each object as a vector    (SigLIP)
  → fuse into one embedding per object across frames
  → text query → cosine similarity → highlight in 3D
```

There is **no per-scene training or optimization** — Lumen3D uses only frozen,
pretrained models. Building a scene is a one-time batch job; *querying* it is
just SigLIP's text encoder plus a cosine sort, so it stays light enough to run
without a GPU.

---

## The three commands

Everything runs through one CLI. **Build** a scene once (GPU), then **query** or
**view** it as many times as you like.

**1. Build a scene** from a video or an image folder:

```bash
lumen3d reconstruct walkthrough.mp4 -o myscene/
```

This writes two files into `myscene/`:

| File | What it is |
|------|------------|
| `bundle.pkl` | the scene bundle — one embedding + 3D points per object (what `query` loads) |
| `scene.ply`  | the full point cloud (what the viewer draws) |

Useful flags: `--stride N` (keep every Nth video frame), `--max-width 1024`
(downscale before the heavy models — mandatory on 8 GB GPUs), `--model`
(DA3 weights), `--conf` (confidence threshold).

**2. Query a built scene** from the terminal:

```bash
lumen3d query myscene/ "a wooden chair"
```

```
Top 5 match(es) for 'a wooden chair':
  1. object  10   score 0.0932   (16609 points)
  2. object  30   score 0.0610   (2463 points)
  ...
```

Scores are low by SigLIP's design — **rank, don't threshold.**

**3. View and search in the browser:**

```bash
lumen3d view myscene/
# open http://127.0.0.1:8000/
```

Orbit the point cloud, type a phrase in the search box, and the matching object
highlights in 3D.

---

## Install

Lumen3D is not on PyPI yet — install from source:

```bash
git clone https://github.com/AwaisAdilKhokhar/lumen3d.git
cd lumen3d
pip install -e .
```

That gives you the `lumen3d` command. The `query` and `view` steps additionally
need `torch` and `uvicorn`; `reconstruct` needs the heavy models
(`depth-anything-3`, `sam2`, `transformers`).

### A note on Python versions (two environments)

Depth Anything 3 pins **Python ≤ 3.13**, so the GPU pipeline can't share a newer
interpreter. This project is developed with two virtual environments:

- **`.venv`** (any recent Python) — runs the test suite and light tooling.
- **`.venv-da3`** (Python 3.12) — holds `torch` + DA3 + SAM 2 + SigLIP, and runs
  `reconstruct`, `query`, and `view`.

On a GPU host, one Python-3.12 environment with all dependencies installed is
enough to run all three commands.

---

## Why it's built this way

- **Frozen models, no optimization loop.** Unlike NeRF/3DGS-style methods,
  nothing is trained per scene. This trades some fidelity for speed and makes the
  whole thing runnable on a free Colab T4.
- **Build is offline; query is light.** Building a scene is GPU-bound and slow;
  answering a query needs only the precomputed per-object embeddings + SigLIP's
  small *text* encoder + cosine. So the demo pre-builds a scene offline and hosts
  only the light query path — a
  [zero-install, no-GPU live demo](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d).
- **Swappable backbone.** The reconstruction stage sits behind a `Backbone`
  interface, so DA3 can be replaced without touching the rest of the pipeline.

---

## Roadmap

- [x] Reconstruction (Depth Anything 3) → point cloud
- [x] Per-frame masks (SAM 2) + per-object embeddings (SigLIP)
- [x] Cross-frame fusion → one embedding per object
- [x] Text query → cosine similarity → 3D highlight
- [x] Web viewer (three.js) with a live search box
- [x] `lumen3d` CLI — `reconstruct` / `query` / `view`
- [x] Colab notebook (build a scene on a free T4)
- [x] Hosted, zero-install live demo (pre-built scene, query-only server)
- [ ] Benchmarks (LERF, Replica) + backbone comparison

---

## License

Apache-2.0.
