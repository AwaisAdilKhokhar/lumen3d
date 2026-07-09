# ---------------------------------------------------------------------------
# Lumen3D hosted demo — the LIGHT, GPU-free query service (FR-10).
#
# Serves ONE pre-built ("frozen") scene: a typed phrase -> SigLIP text vector
# -> cosine rank over the per-instance embedding table -> highlight in 3D.
# It installs NONE of the heavy build stack (Depth Anything 3 / SAM 2): building
# a scene is an offline Colab/GPU job; serving a frozen one needs only numpy,
# SigLIP's text tower, and the three.js viewer. Entry point:
#   src/lumen3d/server.py :: create_demo_app
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# System libs the wheels expect but python:slim omits: libglib2.0-0 (opencv even
# in its headless build) and libgomp1 (OpenMP, used by torch-CPU / opencv).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Fixed HF cache location + import path + which scene to serve. (Offline flags
# are set AFTER the model is baked in, below — setting them now would block the
# prefetch download.)
ENV HF_HOME=/app/hf-cache \
    PYTHONPATH=/app/src \
    LUMEN3D_SCENE=/app/demo \
    PYTHONUNBUFFERED=1

# 1. CPU-only torch + torchvision from PyTorch's CPU wheel index. The default
#    PyPI torch is a ~2 GB CUDA build we never use (no GPU at serve time).
#    torchvision is required by SigLIP's (fast) AutoImageProcessor, which
#    _load_model() constructs even though the hosted path only embeds TEXT.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. The rest of the light serving deps. opencv is the HEADLESS build: server.py
#    imports cv2 at load (via segmentation.py), and python:slim has no system GL
#    libs the GUI build needs.
COPY requirements-host.txt .
RUN pip install --no-cache-dir -r requirements-host.txt

# 3. Bake SigLIP (text tower + tokenizer + image processor) into the image, so
#    the startup warm-up loads from disk and cold starts never hit the network.
RUN python -c "from transformers import AutoModel, AutoTokenizer, AutoImageProcessor; m='google/siglip-base-patch16-224'; AutoModel.from_pretrained(m); AutoTokenizer.from_pretrained(m); AutoImageProcessor.from_pretrained(m)"

# Now that the weights are cached, serve fully offline (no Hub calls, no writable
# cache needed at runtime — important when the host runs the container non-root).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 4. App code (via PYTHONPATH — no pip install: the package's opencv-python dep
#    is the GUI build, and building it needs README.md; both avoided here) and
#    the frozen scene. The viewer front end ships inside the package
#    (src/lumen3d/viewer/), so COPY src/ brings it along.
COPY src/ ./src/
COPY demo/ ./demo/

# HF Spaces convention: serve on 7860, bound to 0.0.0.0.
EXPOSE 7860
CMD ["uvicorn", "lumen3d.server:create_demo_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
