<div align="center">

# 🔦 Lumen3D

### Turn a phone video into a language-queryable 3D scene

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Live Demo](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)** ·
**[Colab Quickstart](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb)** ·
**[How It Works](#-how-it-works)** ·
**[Installation](#-installation)** ·
**[Roadmap](#-roadmap)**

</div>

---

Lumen3D takes a casual video walkthrough (or a set of photos) and produces an
**interactive, language-queryable 3D scene**. Type a phrase — *"the red backpack"*,
*"a wooden chair"* — and the matching object lights up in 3D in your browser.

**No per-scene training. No camera calibration. No workstation GPU.** Lumen3D
composes frozen foundation models — [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
for geometry, [SAM 2](https://github.com/facebookresearch/sam2) for object masks,
[SigLIP](https://huggingface.co/google/siglip-base-patch16-224) for open-vocabulary
semantics — into one pipeline that builds a scene on a **free Colab T4** and queries
it on a CPU.

<div align="center">

<video src="https://github.com/AwaisAdilKhokhar/lumen3d/raw/main/docs/demo.mp4" controls muted loop width="720"></video>

<sub><i>Typing a phrase lights up the matching object in 3D — no per-scene training, running on a CPU.</i></sub>

<sub>▶️ <a href="https://github.com/AwaisAdilKhokhar/lumen3d/raw/main/docs/demo.mp4">Watch the demo video</a> · 🔦 <a href="https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d">Try it live</a></sub>

</div>

> 🚧 **Status:** v0.1 in active development, built in public. Core pipeline, CLI,
> web viewer, Colab notebook, and hosted demo are done; benchmarks are in progress.

---

## ✨ Highlights

- **🗣️ Open-vocabulary 3D search** — free-text queries resolve to 3D object
  instances via SigLIP embeddings + cosine ranking. No fixed label set.
- **⚡ Zero per-scene optimization** — unlike the LERF/LangSplat line, nothing is
  trained per scene. Reconstruction is a single feed-forward pass, so the full
  pipeline fits a free Colab T4.
- **🧩 Object instances, not a blob** — objects are discovered on *every* frame,
  then associated in 3D (voxel-overlap **and** embedding-similarity gates, à la
  ConceptGraphs), so late-appearing objects are found and adjacent ones stay separate.
- **🌐 Build heavy, query light** — building a scene is a one-time GPU batch job;
  querying needs only the precomputed per-instance embedding table + SigLIP's tiny
  text encoder. That split is what makes the zero-install hosted demo possible.
- **🔌 Swappable backbone** — reconstruction sits behind a `Backbone` interface, so
  DA3 can be exchanged for VGGT/π³ without touching the rest of the pipeline.
- **🛠️ Tool, not research code** — pip-installable package, 3-command CLI, three.js
  web viewer, Colab notebook, and a test suite (50+ tests, no GPU required).

---

## 🔦 Try It Now

**[Open the live demo →](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)** —
explore a real reconstructed apartment and search it by typing a phrase. No install,
no GPU, no build step.

> Runs on a free CPU Space: it sleeps after inactivity (first load may take ~30 s to
> wake), and the showcase scene is a large point cloud, so give it a moment to render.

Want to build a scene from **your own video**? Use the
[Colab notebook](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb) —
upload a clip, build on a free T4, download the scene, and view it locally.

---

## ⚙️ How It Works

```
Video / photos
  → pick good frames                     (keyframe sampling)
  → reconstruct a 3D point cloud         (Depth Anything 3)
  → outline objects on EVERY frame       (SAM 2)
  → embed each detection as a vector     (SigLIP)
  → associate detections in 3D           (voxel overlap + embedding similarity)
  → one embedding + point cloud per object instance
  → text query → cosine similarity → highlight in 3D
```

Every learned component is **pretrained and frozen** — the scene-specific work is
pure geometry: unprojection, voxel-grid association, and embedding bookkeeping.
This is the deliberate contrast with optimization-based language-3D methods
(LERF, LangSplat, OpenGaussian), which spend minutes-to-hours of GPU time per scene.

|  | Optimized language-3DGS<br>(LangSplat, OpenGaussian) | Raw feed-forward tools<br>(DA3/VGGT CLI) | **Lumen3D** |
|---|:---:|:---:|:---:|
| Per-scene training | Required (slow) | None | **None** |
| Language querying | Yes | No | **Yes** |
| Object instances | Sometimes | No | **Yes** |
| Hardware | High-end GPU | Modest | **Free Colab T4** |
| Packaging / DX | Research code | CLI only | **pip + CLI + viewer + hosted demo** |

---

## 🚀 Usage

Everything runs through one CLI. **Build** a scene once (GPU), then **query** or
**view** it as many times as you like (CPU).

### 1. Build a scene

```bash
lumen3d reconstruct walkthrough.mp4 -o myscene/
```

Accepts a video file or an image folder. Writes two files into `myscene/`:

| File | What it is |
|------|------------|
| `bundle.pkl` | the scene bundle — one embedding + 3D points per object (what `query` loads) |
| `scene.ply`  | the full point cloud (what the viewer draws) |

Useful flags: `--stride N` (keep every Nth video frame), `--max-width 1024`
(downscale before the heavy models — mandatory on 8 GB GPUs), `--model`
(DA3 weights), `--conf` (confidence threshold).

### 2. Query from the terminal

```bash
lumen3d query myscene/ "a wooden chair"
```

```
Top 5 match(es) for 'a wooden chair':
  1. object  10   score 0.0932   (16609 points)
  2. object  30   score 0.0610   (2463 points)
  ...
```

Scores are low by SigLIP's sigmoid design — **rank, don't threshold.**

### 3. View and search in the browser

```bash
lumen3d view myscene/
# open http://127.0.0.1:8000/
```

Orbit the point cloud, type a phrase in the search box, and the matching object
highlights in 3D.

---

## 📦 Installation

Lumen3D is not on PyPI yet — install from source:

```bash
git clone https://github.com/AwaisAdilKhokhar/lumen3d.git
cd lumen3d
pip install -e .
```

That gives you the `lumen3d` command. On top of the base install:

- `reconstruct` needs the heavy stack — `torch` (CUDA), `depth-anything-3`, `sam2`.
- `query` and `view` need only `torch` (CPU is fine) and `uvicorn`.

> **Python version note:** Depth Anything 3 pins **Python ≤ 3.13**, so use a
> Python 3.12 environment for the full pipeline. (This repo is developed with a
> light 3.14 venv for the test suite and a 3.12 venv for the GPU stack; on a GPU
> host, one 3.12 environment with everything installed is all you need.)

**No GPU?** Build scenes on the free-tier
[Colab notebook](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb),
then download and `lumen3d view` them locally.

---

## 📊 Benchmarks

Coming with v0.1: open-vocabulary localization accuracy and mIoU on **LERF** and
**Replica**, plus a backbone comparison (DA3 vs VGGT vs π³) — the story being
*competitive open-vocab querying with zero per-scene optimization, on commodity
hardware*.

| Backbone | Localization Acc. ↑ | mIoU ↑ | End-to-end time ↓ | Peak GPU mem ↓ |
|---|:---:|:---:|:---:|:---:|
| DA3 (default) | *in progress* | — | — | — |
| VGGT | — | — | — | — |
| π³ | — | — | — | — |

---

## 🗺️ Roadmap

- [x] Feed-forward reconstruction (Depth Anything 3) → point cloud
- [x] Per-frame object discovery (SAM 2) + region embeddings (SigLIP)
- [x] 3D instance association (voxel overlap + semantic merge gate)
- [x] Text query → cosine similarity → 3D highlight
- [x] Web viewer (three.js) with a live search box
- [x] `lumen3d` CLI — `reconstruct` / `query` / `view`
- [x] Colab notebook (build a scene on a free T4)
- [x] Hosted, zero-install live demo (pre-built scene, query-only server)
- [ ] Benchmarks (LERF, Replica) + backbone comparison
- [ ] `.glb` export
- [ ] **Phase 2:** VLM spatial QA (*"how many chairs?"*) · optional 3DGS refinement pass

---

## 🙏 Acknowledgments

Lumen3D stands on excellent open work:

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — feed-forward multi-view depth + pose
- [SAM 2](https://github.com/facebookresearch/sam2) — promptable image/video segmentation
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) — language-aligned image/text embeddings
- [three.js](https://threejs.org/) — the browser viewer
- [ConceptGraphs](https://concept-graphs.github.io/) — the geometric + semantic instance-association recipe
- [LERF](https://www.lerf.io/) / [LangSplat](https://langsplat.github.io/) — the optimization-based language-3D line this project contrasts with

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).
