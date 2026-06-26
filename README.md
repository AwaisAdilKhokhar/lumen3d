# Lumen3D

> Turn a phone video into a language-queryable 3D scene — on free Colab.

Lumen3D takes a casual video walkthrough (or a set of photos) and produces an
interactive, language-queryable 3D scene. Type a phrase — *"the red backpack"*,
*"every chair"* — and the matching regions light up in 3D in your browser.

**Status:** 🚧 Early development (v0.1). Built in public as a learning + portfolio project.

## How it works

```
Video / photos
  → pick good frames
  → reconstruct a 3D point cloud   (Depth Anything 3)
  → outline objects per frame      (SAM2)
  → embed each object as numbers   (SigLIP)
  → fuse into per-instance embeddings
  → text query → cosine similarity → highlight in 3D
```

No per-scene training or optimization — only frozen pretrained models. That's what
lets the whole pipeline run on a free Colab T4.

## Install

```bash
# coming soon
pip install lumen3d
```

## License

Apache-2.0.
