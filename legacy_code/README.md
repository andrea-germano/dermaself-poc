# Legacy pipeline — phase 2 + phase 3 variants

This folder contains the **legacy implementation** of the anonymization pipeline after 3DDFA‑V3 reconstruction. It includes:

- **Phase 2 – bundle & segmentation utilities**
- **Phase 3 – three alternative anonymizers** that take Phase‑2 bundles and produce morphed meshes + re‑baked textures.

Use this when you need the older, more granular flow or to reproduce past experiments.

---

## Prerequisites
- You already ran **face reconstruction** (3DDFA‑V3) and have, for each input image, an extracted textured OBJ such as `<stem>_extractTex.obj` plus its PNG texture in per‑image subfolders. (See the main project README for step‑1.)
- Python environment matches the main repo (NumPy, OpenCV, SciPy, PyTorch if used by scripts, etc.).

> Tip: all Phase‑3 variants expect the **Phase‑2 bundle structure** described below.

---

## Phase 2 — Build anonymization bundles
**Script:** `fase2_segm.py`

**What it does**
- Scans 3DDFA‑V3 results (recursively) to find `*_extractTex.obj` and their textures.
- Parses the OBJ+UV, computes UV resolution (or uses `--uv_size`) and writes a **bundle** per image under `--out_root/<stem>/` containing:
  - `mesh/verts.npy`, `mesh/faces.npy`, `mesh/uv_coords.npy`
  - `mesh/u_mean.npy` (mean face), `mesh/procrustes.json` (alignment)
  - `masks_uv/*.png` (UV masks),
  - optional `mesh/trans_params.npy` if `--trans_params_root` is provided,
  - a copy/reference to the source texture and image metadata.

**CLI**
```bash
python fase2_segm.py `
  --input_root examples/results `
  --assets_dir assets `
  --out_root stage3_bundles `
  --images_root path/to/original_images `
  --trans_params_root path/to/trans_params_npy_root `
  --uv_size 0 `
  --flip_v 0
```
- `--input_root` root with the per‑image subfolders produced by Step‑1.
- `--uv_size 0` means: infer from the PNG; otherwise set a fixed value (e.g., 2048).
- `--trans_params_root` lets the script copy `<stem>_trans_params.npy` if present.

**Output**
- Bundles saved under `--out_root` (default `stage3_bundles/`).

---

## Phase 3 — Anonymization (choose one variant)
All Phase‑3 scripts read **Phase‑2 bundles** from `--bundles_dir` and write morphed meshes + re‑baked textures under `--out_dir`.

### Variant A — SLIM (few knobs, quality‑preserving)
**Script:** `fase3_arap_sempl.py`

**Concept**
- ARAP with per‑vertex soft constraints and smoothing; focus factor to emphasize facial features; UV re‑bake per triangle; optional eye‑black.

**CLI**
```bash
python fase3_arap_sempl.py `
  --bundles_dir stage3_bundles `
  --out_dir stage3_morphed_slim `
  --strength 1.0 `
  --focus 1.8 `
  --rebake_res 4096 `
  --sharpen 1 `
  --eye_black 1 `
  --eye_dark_value 0 `
  --flip_v 0
```
**Outputs**
- `mesh/verts_morphed.npy`, updated OBJ/MTL, re‑baked PNG texture in `--out_dir/<stem>/`.

---

### Variant B — ARAP v5 (feature‑aware + tunable solver)
**Script:** `fase3_arap_v5.py`

**Concept**
- Builds continuous local/global targets with per‑vertex weights (eyes, lips, nose, brows), optional **auto‑strength** scaled by face size, then solves with **harmonic / biharmonic / ARAP** (default ARAP). Re‑bakes UV and can darken eyes.

**CLI**
```bash
python fase3_arap_v5.py `
  --bundles_dir stage3_bundles `
  --out_dir stage3_morphed `
  --beta_local 0.90 --beta_global 0.35 `
  --t_low 0.10 --t_high 0.85 `
  --mouth_gain 1.50 --eyes_gain 1.25 --nose_gain 1.15 --brows_gain 1.00 `
  --focus_power 1.8 --global_inside 0.55 --global_outside 0.12 `
  --solver arap --arap_w 45.0 --arap_iters 10 `
  --smooth_disp_alpha 0.12 --auto_strength 1 --anon_ratio 0.085 `
  --rebake 1 --rebake_uv_res 2048 --sharpen 1 --eye_black 1 --flip_v 0
```
**Outputs**
- Morphed mesh (`verts_morphed.npy`) + OBJ/MTL with UV; re‑baked texture PNG.

---

### Variant C — FFD (grid‑based free‑form deformation)
**Script:** `fase3_ffd.py`

**Concept**
- 3D FFD with class‑weighted focus (eyes/nose/mouth/temples/cheeks/chin/face), asymmetry control, optional high‑res UV re‑bake.

**CLI**
```bash
python fase3_ffd.py `
  --bundles_dir stage3_bundles `
  --out_dir stage3_morphed `
  --strength 1.2 --class_var 0.7 --asym 0.3 `
  --grid 9x11x9 --seed 0 `
  --uv_res 3072 --flip_v 0 `
  --w_eyes 1.0 --w_nose 1.0 --w_mouth 1.0 `
  --w_temples 0.30 --w_cheeks 0.10 --w_chin 0.10 --w_face 0.00
```

---

## Typical end‑to‑end (legacy)
1) **Build bundles** from 3DDFA‑V3 outputs:
```bash
python fase2_segm.py --input_root examples/results --assets_dir assets --out_root stage3_bundles `
  --images_root data/images --uv_size 0 --flip_v 0
```
2) **Choose ONE Phase‑3 variant**, e.g., SLIM:
```bash
python fase3_arap_sempl.py --bundles_dir stage3_bundles --out_dir stage3_morphed_slim `
  --strength 1.0 --focus 1.8 --rebake_res 4096 --sharpen 1 --eye_black 1
```
(or run the ARAP v5 / FFD commands above.)

---

## Notes & caveats
- All Phase‑3 re‑bakes **sample from the ORIGINAL mesh** when generating the new UV texture to minimize drift; the MTL is rewritten to point to the rebaked PNG.
- Use `--flip_v 1` if your downstream tool treats OBJ `vt` with inverted V.
- For large faces/textures, prefer higher re‑bake resolutions (e.g., 4096) on SLIM/ARAP v5.

---

## Related utilities (optional)
- `segmentation.py`: creates RGBA cutouts removing eyes (and optionally mouth) guided by `seg_visible_labels_full` maps, with feathering/inpaint.
```bash
python segmentation.py `
  --in_root <3ddfa_results_root> `
  --out_root <out_png_root> `
  --images_root <original_images_root> `
  --remove_mouth 1 --margin_px 0 --feather_px 1.0
```
Outputs a single `<stem>_masked.png` per subfolder.

---

## Troubleshooting
- **\"PNG texture missing\"** during Phase‑3: ensure each bundle contains or can reference the original PNG texture exported next to `*_extractTex.obj`.
- **\"No bundles found\"**: check `--out_root` from Phase‑2 and pass it as `--bundles_dir` to Phase‑3.
