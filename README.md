<div align="center">

# Lumen3D

### Turn a phone video into a language-queryable 3D scene

[![PyPI](https://img.shields.io/pypi/v/lumen3d)](https://pypi.org/project/lumen3d/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Live Demo](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)** ·
**[Colab Quickstart](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb)** ·
**[How It Works](#how-it-works)** ·
**[Installation](#installation)**

</div>

---

Lumen3D takes a casual video walkthrough (or a set of photos) and produces an
interactive 3D scene you can search with plain language. Type a phrase —
*"the red backpack"*, *"a wooden chair"* — and the matching object lights up in
your browser.

There is no per-scene training, no camera calibration, and no need for a
workstation GPU. Lumen3D composes three frozen foundation models —
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for
geometry, [SAM 2](https://github.com/facebookresearch/sam2) for object masks,
and [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) for
open-vocabulary semantics — into a single pipeline. Everything scene-specific
is geometry and bookkeeping, which is why a scene builds on a free Colab T4 and
queries on a plain CPU.

<div align="center">

<img src="https://github.com/AwaisAdilKhokhar/lumen3d/raw/main/assets/demo.gif" alt="Lumen3D demo — orbiting a reconstructed apartment and searching it by text" width="800" />

<sub><i>Typing a phrase lights up the matching object in 3D — no per-scene training, running on a CPU.</i></sub>

<sub><a href="https://github.com/AwaisAdilKhokhar/lumen3d/raw/main/assets/demo.mp4">Watch the full demo video</a> · <a href="https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d">Try it live</a></sub>

</div>

> **Status:** v0.1 in active development, built in public. The core pipeline,
> CLI, web viewer, Colab notebook, and hosted demo are done.

---

## Try it

**[Open the live demo](https://huggingface.co/spaces/AwaisAdilKhokhar/lumen3d)** —
explore a real reconstructed apartment and search it by typing a phrase. No
install, no GPU, no build step. It runs on a free CPU Space, so it sleeps after
inactivity and the first load may take about 30 seconds to wake up.

To build a scene from your own video, use the
[Colab notebook](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb):
upload a clip, build on a free T4, download the scene, and view it locally.

---

## How it works

```
Video / photos
  → pick good frames                     (keyframe sampling)
  → reconstruct a 3D point cloud         (Depth Anything 3)
  → outline objects on every frame       (SAM 2)
  → embed each detection as a vector     (SigLIP)
  → associate detections in 3D           (voxel overlap + embedding similarity)
  → one embedding + point cloud per object instance
  → text query → cosine similarity → highlight in 3D
```

Every learned component is pretrained and frozen. This is a deliberate contrast
with the optimization-based language-3D line (LERF, LangSplat, OpenGaussian),
which distills language features into a radiance field or Gaussian splat and
spends minutes to hours of GPU time per scene. Lumen3D instead does
open-vocabulary querying directly on a cheap feed-forward reconstruction — the
scene-specific work is pure geometry, so it fits on commodity hardware.

|  | Optimized language-3DGS (LangSplat, OpenGaussian) | Raw feed-forward tools (DA3/VGGT CLI) | Lumen3D |
|---|:---:|:---:|:---:|
| Per-scene training | Required (slow) | None | None |
| Language querying | Yes | No | Yes |
| Object instances | Sometimes | No | Yes |
| Hardware | High-end GPU | Modest | Free Colab T4 |
| Packaging | Research code | CLI only | pip + CLI + viewer + hosted demo |

### Reconstruction: from frames to a point cloud

Classically, recovering 3D structure from images meant structure-from-motion
(COLMAP) followed by per-scene optimization (a NeRF or Gaussian splat).
Depth Anything 3 belongs to a newer family of feed-forward models: given a
handful of frames, a single forward pass predicts a depth map for every frame
*plus* the camera intrinsics and poses that relate them. No calibration, no
optimization loop.

A depth map alone lives in its own camera's coordinates. To get a shared scene,
each pixel is *unprojected*: a pixel at `(u, v)` with depth `d` back-projects
along its camera ray (the inverse intrinsics applied to the homogeneous pixel,
scaled by `d`), and the camera-to-world transform then places that point in a
coordinate frame common to all views. Repeat for every valid pixel of every
frame and the depth maps fuse into one point cloud. Lumen3D implements this
unprojection itself in plain NumPy (`src/lumen3d/geometry.py`) and verifies it
point-for-point against DA3's internal helper in the test suite.

One property matters downstream: like most feed-forward reconstructors, DA3's
output is *up to scale* — geometrically consistent, but not in meters. Any
threshold expressed in absolute units would be meaningless, which shapes how
instance association is done below.

### Semantics: what makes a scene queryable by text

SigLIP is a contrastive image–text model in the CLIP family: an image encoder
and a text encoder trained jointly so that an image and a caption that belong
together land near each other in a shared embedding space. That shared space is
the entire trick behind open-vocabulary search — there is no fixed label set,
because any phrase the text encoder can process becomes a point in the same
space the image crops live in.

For every object mask SAM 2 proposes, Lumen3D takes the tight rectangular crop
around the mask (background included — SigLIP was trained on natural photos,
and a blacked-out cutout is out-of-distribution) and embeds it into a 768-d
vector. A text query goes through the text encoder into the same space, and
matching is a cosine similarity against the per-object table followed by a
sort. That is one small matrix multiply, which is why querying needs no GPU.

Two quirks of the embedding space are worth knowing. SigLIP is trained with a
sigmoid loss rather than a softmax, so raw similarity scores are small
(typically 0.05–0.15 for a correct match) and should never be thresholded —
Lumen3D ranks instead. And CLIP-style spaces are anisotropic: even unrelated
image pairs sit at a high baseline cosine (around 0.8), so what carries signal
is the *gap* above that baseline, not the absolute value.

### Instance association: turning detections into objects

SAM 2's automatic mask generator runs on every frame, not just the first. That
means an object revealed halfway through the video still gets discovered — but
it also means the same physical chair, seen from six angles, starts life as six
unrelated detections. Association has to merge those into one instance without
also fusing genuinely different objects that happen to sit next to each other.

Lumen3D resolves identity in 3D, following the recipe from
[ConceptGraphs](https://concept-graphs.github.io/). Each detection is
unprojected through its own mask into a world-space fragment, and fragments are
snapped onto a voxel grid whose cell size is a fraction of the scene's
bounding-box diagonal — a relative measure, so it survives DA3's arbitrary
scale. Two detections merge only if they pass **both** gates:

1. **Geometric:** their voxel sets overlap (voxel IoU above a threshold) —
   they occupy the same region of space.
2. **Semantic:** their SigLIP embeddings agree (cosine above a threshold) —
   they look like the same kind of thing.

Neither gate is sufficient alone. Geometry by itself mis-merges adjacent
objects — an early version of this pipeline confidently fused a wooden door
with the trash can beside it, because they shared enough voxels. Semantics by
itself would merge two identical chairs across the room. Requiring both keeps
the door and the trash can separate while still fusing the chair across
viewpoints. An instance's final embedding is the mean over its member
detections, and its geometry is the union of their points.

The result is one embedding and one point cloud per object instance. This is
the other deliberate departure from LERF/LangSplat, which attach language
features to the field itself (per point or per Gaussian). Per-point features
suffer "feature bleed" — semantics smearing across object boundaries, so a
query for "mug" softly lights up half the desk — and cost memory proportional
to the scene. One vector per instance gives crisp boundaries and collapses a
scene's semantic footprint to a table of a few hundred 768-d vectors.

### Build heavy, query light

Building a scene runs DA3, SAM 2, and SigLIP over dozens of frames — minutes of
GPU time. But the finished scene is a frozen bundle: a point cloud plus that
small embedding table. Answering a query against it needs only SigLIP's text
encoder and a cosine, so the hosted demo runs on a free CPU tier and never
imports DA3 or SAM 2 at all. A test in the suite enforces this boundary: it
builds the query server in a fresh interpreter and asserts the heavy packages
were never imported, even in an environment where they are installed.

---

## Usage

Everything runs through one CLI. Build a scene once (GPU), then query or view
it as many times as you like (CPU).

### 1. Build a scene

```bash
lumen3d reconstruct walkthrough.mp4 -o myscene/
```

Accepts a video file or an image folder. Writes two files into `myscene/`:
`bundle.pkl` (one embedding + 3D points per object — what `query` loads) and
`scene.ply` (the full point cloud — what the viewer draws).

Useful flags: `--stride N` (keep every Nth video frame), `--max-width 1024`
(downscale before the heavy models — necessary on 8 GB GPUs), `--model`
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

Scores are low by SigLIP's sigmoid design — rank, don't threshold.

### 3. View and search in the browser

```bash
lumen3d view myscene/
# open http://127.0.0.1:8000/
```

Orbit the point cloud, type a phrase in the search box, and the matching object
highlights in 3D.

---

## Installation

```bash
pip install lumen3d
```

That gives you the `lumen3d` command, ready to `view` and `query` scenes
(the base install includes CPU-capable `torch` and the web viewer). Or install
from source:

```bash
git clone https://github.com/AwaisAdilKhokhar/lumen3d.git
cd lumen3d
pip install -e .
```

On top of the base install, `reconstruct` (building scenes) needs the heavy
stack — `torch` (CUDA), `depth-anything-3`, `sam2`.

> **Python version note:** Depth Anything 3 pins Python ≤ 3.13, so use a
> Python 3.12 environment for the full pipeline. On a GPU host, one 3.12
> environment with everything installed is all you need.

No GPU? Build scenes on the free-tier
[Colab notebook](https://colab.research.google.com/github/AwaisAdilKhokhar/lumen3d/blob/main/notebooks/lumen3d_colab.ipynb),
then download and `lumen3d view` them locally.

---

## Acknowledgments

Lumen3D stands on excellent open work:

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — feed-forward multi-view depth + pose
- [SAM 2](https://github.com/facebookresearch/sam2) — promptable image/video segmentation
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) — language-aligned image/text embeddings
- [three.js](https://threejs.org/) — the browser viewer
- [ConceptGraphs](https://concept-graphs.github.io/) — the geometric + semantic instance-association recipe
- [LERF](https://www.lerf.io/) / [LangSplat](https://langsplat.github.io/) — the optimization-based language-3D line this project contrasts with

## License

This project is released under the [Apache 2.0 license](LICENSE).
