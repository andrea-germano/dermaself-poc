# 3D Face Anonymization Pipeline (powered by 3DDFA‑V3)

This repository runs an **end‑to‑end** pipeline to anonymize faces from 2D images:
1. **Face 3D** – reconstruct a 3D face and auxiliary projections (custom version pof the open project 3DDFA‑V3).
2. **Segmentation** – produce an RGBA mask that removes the eyes (and optionally the mouth) from the original image.
3. **Anonymization** – morph the mesh toward a mean face + re‑bake a high‑resolution texture.
4. **Renderer** – render the anonymized 3D object to a 2D image (PyTorch3D).

> Note: **nvdiffrast is not required**. Reconstruction uses the **CPU renderer** provided by 3DDFA‑V3 (Cython).

---

## Requirements & Setup

### 1) Clone and create the environment
```bash
git clone https://github.com/andrea-germano/dermaself-poc.git
cd dermaself-poc

# Recommended: conda (Windows/Linux/macOS)
conda create -n DDFAV3-anon python=3.8 -y
conda activate DDFAV3-anon

# PyTorch (tested version)
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
# Alternatively (Windows/Linux): follow the selector on pytorch.org
```

### 2) Python dependencies
```bash
pip install -r requirements.txt
```

### 3) Build the **CPU** renderer (Cython)
```bash
cd ddfa_v3/util/cython_renderer/
python setup.py build_ext -i
cd ../../..
```
> Windows: if you get *“Microsoft Visual C++ 14.x is required”*, install **Visual Studio Build Tools** (C++ components & Windows SDK), reopen the terminal, and rebuild.

### 4) (Optional but useful) **PyTorch3D** for 2D rendering
Only needed for **Step 4 – Renderer** (to save a PNG). If the **.obj** from Step 3 is enough, skip this.
```bash
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
>Make sure you have build tools (e.g., `build-essential in linux`).

### 5) Assets & weights
Download the files required by 3DDFA‑V3 into `ddfa_v3/assets/` (models, landmark indices, etc.).
- For **Step 3 – Anonymization** you need **`face_model.npy` (or `.npz`)** with the key **`'u'`** (mean face coordinates).
- See `ddfa_v3/assets/README.md` for links to the weights.

---

## Suggested folder layout
```
dermaself-poc/
├─ input/                 # input images (.png/.jpg) — unique names
├─ out/
│  ├─ face3d/            # step 1 — one subfolder per image
│  ├─ seg/               # step 2 — RGBA PNGs with eyes/mouth removed
│  ├─ anon/              # step 3 — OBJ + anonymized texture
│  └─ render/            # step 4 — rendered PNGs
└─ ddfa_v3/              # 3DDFA‑V3 code + assets/
```

---

## Step‑by‑step execution

### Step 1 — Face 3D (reconstruction with 3DDFA‑V3)
Script: `face_3d.py`

Example (CPU):
```bash
python face_3d.py \
  --inputpath input/ \
  --savepath out/face3d \
  --device cpu \
  --iscrop 1 \
  --detector retinaface \
  --ldm68 1 --ldm106 1 --ldm106_2d 1 --ldm134 1 \
  --seg_visible 1 --seg 1 \
  --useTex 0 --extractTex 1 \
  --backbone resnet50
```
- **Input**: folder with images (`--inputpath`).  
- **Output**: `--savepath` will contain **one subfolder per image** (based on the file stem) with:
  - `*_extractTex.obj` – triangulated mesh with UV extracted from the image;
  - `*_projections.npz` – precomputed projections/segmentations. Required keys:
    - `seg_visible_labels_full` (for Step 2);
    - `v2d_full` (for Step 3).
- **Detector**: `retinaface` (default) or `mtcnn`. If images are already aligned/cropped at 224×224, you can use `--iscrop 0`.

> If you want to use GPU for the **model** (not for the renderer), set `--device cuda` (requires a CUDA build of PyTorch).

---

### Step 2 — Segmentation (create RGBA mask)
Script: `segmentation.py`

Example:
```bash
python segmentation.py \
  --in_root out/face3d \
  --images_root input \
  --out_root out/seg \
  --remove_mouth 1 \
  --margin_px 6 \
  --feather_px 1.5
```
- **Input**: `--in_root` = Step 1 output with **one subfolder per image**.  
  For each subfolder it reads `<stem>_projections.npz` and key `seg_visible_labels_full`.
- **images_root**: folder with the **original images**, flat (not recursive) — it matches `<stem>.png|jpg|jpeg|bmp`.  
  Image size must match the one used for projections.
- **Output**: one RGBA PNG per image in `--out_root`, named `"<stem>_masked.png"`:  
  - RGB with **inpaint** in the removed areas;
  - Alpha=0 over removed areas; **feather** controls border smoothness.

Key options:
- `--remove_mouth 1` also removes the mouth (labels 6,7). Use `0` to keep the mouth.
- `--margin_px` adds dilation (px) around eyes/mouth.
- `--feather_px` = Gaussian sigma for soft transitions (0 disables).

---

### Step 3 — Anonymization (morph + re‑bake UV texture)
Script: `anonymization.py`

Example:
```bash
python anonymization.py \
  --in_root out/face3d \
  --images_root input \
  --face_model ddfa_v3/assets/face_model.npy \
  --out_root out/anon \
  --alpha 1.0 \
  --tex_size 2048 \
  --keep_pose
```
- **Input** (for each subfolder in `--in_root`):
  - `*_extractTex.obj` (mesh with UV);
  - `<stem>_projections.npz` with **`v2d_full`** (per‑vertex 2D projection);
  - matching original image from `--images_root`.
- **Output** per image in `--out_root/<subfolder>`:
  - `<stem>.obj` – **anonymized mesh** (morphed toward the mean face);
  - `<stem>_texture.png` – **rebaked UV texture** at high resolution.
- `--face_model` must contain the key `'u'` (mean face, 3N values).  
- `--alpha` (0..1) controls morph strength (1.0 = full mean face).  
- `--keep_pose` preserves the global pose (Umeyama); drop it to normalize the pose as well.
- `--tex_size` (e.g., 1024, 2048, 4096) sets the UV texture resolution.

> The script **stops** on the first error (missing files, shape mismatch, …) to surface prerequisites early.

---

### Step 4 — Renderer (PNG from the anonymized model) — **CLI‑based**
Script: `renderer.py` (uses **PyTorch3D**).

> This is the heaviest step. You can render **a single .obj** or **all subfolders** in `--in_root` (like in other steps).

**A) Single .obj file**
```bash
python renderer.py \
  --obj out/anon/edo/edo.obj \
  --out out/render/edo_front.png \
  --img_size 1024 \
  --margin 0.95 \
  --bg_white 0 \
  --device cpu
```
- `--bg_white 1` saves **RGB** composited on white. With `0` you get **PNG RGBA**.
- `--margin` (0..1) controls how much of the frame the mesh fills.
- `--alpha_thresh` (default 0.5) sharpens/softens the binary alpha edges.

**B) Batch over all subfolders**
```bash
python renderer.py \
  --in_root out/anon \
  --out_root out/render \
  --img_size 1024 \
  --margin 0.95 \
  --bg_white 0 \
  --device cpu
```
- For each subfolder in `out/anon`, the script first looks for **`<subdir>/<subdir>.obj`**; if not found, it falls back to the **first `.obj`** in the folder.
- Output is saved as `out/render/<subdir>_front.png`.
- Use `--stop_on_error` to stop at the first error.

> On macOS/Apple Silicon, **PyTorch3D** does not provide GPU (MPS) support; CPU‑only works but is slower. Alternatively, import the `.obj` + **texture** from Step 3 into a DCC (e.g., Blender) and render there.

---

## Tips & Troubleshooting

- **`ModuleNotFoundError: Cython`**: install `cython` in your env and rebuild the Cython extension (`setup.py build_ext -i`).
- **C/C++ build on Windows**: install *Microsoft C++ Build Tools* (MSVC 14.x) and the *Windows 10/11 SDK*.
- **Missing `*_projections.npz`**: rerun Step 1 with `--seg_visible 1 --extractTex 1`.
- **Image/segmentation size mismatch**: Step 2 requires the **same resolution** as the image used in Step 1 (same `<stem>`).

---

## Licenses & credits
- 3D reconstruction is based on **3DDFA‑V3** (academic). This repo integrates anonymization components and rendering tools.
