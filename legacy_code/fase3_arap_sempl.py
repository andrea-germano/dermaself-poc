import json
from pathlib import Path
import argparse
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================== Util =========================================
def smootherstep01(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0).astype(np.float32)
    return t*t*t*(t*(t*6.0 - 15.0) + 10.0)

def robust_percentile(x, p=90.0):
    x = np.asarray(x, np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    p = np.clip(float(p), 0.0, 100.0)
    return float(np.percentile(np.abs(x), p))

def cotmatrix(V, F):
    V = V.astype(np.float64); F = F.astype(np.int64)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]
    e0 = v1 - v2; e1 = v2 - v0; e2 = v0 - v1
    def _cot(a, b):
        num = (a*b).sum(axis=1)
        den = np.linalg.norm(np.cross(a, b), axis=1) + 1e-12
        return num / den
    cot0 = _cot(e1, e2); cot1 = _cot(e2, e0); cot2 = _cot(e0, e1)
    I = np.concatenate([i1, i2, i0, i2, i0, i1])
    J = np.concatenate([i2, i1, i2, i0, i1, i0])
    W = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    n = V.shape[0]
    W = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    d = np.array(W.sum(axis=1)).ravel()
    return sp.diags(d) - W

def mass_matrix_lumped(V, F):
    Vd = V.astype(np.float64); F = F.astype(np.int64)
    v0, v1, v2 = Vd[F[:, 0]], Vd[F[:, 1]], Vd[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    n = V.shape[0]
    M = sp.lil_matrix((n, n))
    for k, (a, b, c) in enumerate(F):
        w = A[k] / 3.0
        M[a, a] += w; M[b, b] += w; M[c, c] += w
    return M.tocsr()

def diffuse_vertex_scalar(V, F, w0, tau=0.40, iters=12):
    L = cotmatrix(V, F)
    d = np.maximum(L.diagonal(), 1e-12)
    W = sp.diags(d) - L
    w = np.asarray(w0, np.float64).reshape(-1)
    for _ in range(int(iters)):
        neigh = W.dot(w) / d
        w = (1.0 - tau) * w + tau * neigh
    return np.clip(w, 0.0, 1.0).astype(np.float32)

def screened_poisson_smooth(V, F, disp, alpha=0.12):
    L = cotmatrix(V, F)
    M = mass_matrix_lumped(V, F)
    A = (M - float(alpha) * L).tocsr()
    B = M.dot(disp.astype(np.float64))
    X = np.empty_like(disp, dtype=np.float64)
    for k in range(3):
        X[:, k] = spla.spsolve(A, B[:, k])
    return X.astype(np.float32)

def feather_mask(mask_u8, band_px=28):
    m = (mask_u8.astype(np.uint8) > 0).astype(np.uint8)
    if band_px <= 0: return m.astype(np.float32)
    dist_out = cv2.distanceTransform((1 - m) * 255, cv2.DIST_L2, 3).astype(np.float32)
    dist_in  = cv2.distanceTransform(m * 255,     cv2.DIST_L2, 3).astype(np.float32)
    sdf = dist_out - dist_in
    k = max(1.0, float(band_px) * 0.75)
    w = 0.5 * (1.0 - np.tanh(sdf / k)).astype(np.float32)
    return np.clip(w, 0.0, 1.0)

def _bilinear_uv_sample(UV: np.ndarray, img01: np.ndarray) -> np.ndarray:
    H, W = img01.shape[:2]
    u = np.clip(UV[:, 0] * (W - 1), 0, W - 1)
    v = np.clip(UV[:, 1] * (H - 1), 0, H - 1)
    x0 = np.floor(u).astype(np.int32); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(v).astype(np.int32); y1 = np.clip(y0 + 1, 0, H - 1)
    Ia = img01[y0, x0]; Ib = img01[y1, x0]; Ic = img01[y0, x1]; Id = img01[y1, x1]
    wa = (x1 - u) * (y1 - v); wb = (x1 - u) * (v - y0)
    wc = (u - x0) * (y1 - v); wd = (u - x0) * (v - y0)
    return np.clip(wa*Ia + wb*Ib + wc*Ic + wd*Id, 0.0, 1.0).astype(np.float32)

def _project_to_crop224(V_obj):
    V_cam = V_obj.astype(np.float32).copy()
    V_cam[:, 2] = 10.0 - V_cam[:, 2]
    f, cx, cy = 1015.0, 112.0, 112.0
    P = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    proj = V_cam @ P.T
    return (proj[:, :2] / proj[:, 2:3])  # [0,224)

def _back_resize_ldms(ldms, trans_params):
    w0, h0, s, tx, ty = trans_params
    target = 224.0
    w = int(w0 * s); h = int(h0 * s)
    left = int(w / 2 - target / 2 + (tx - w0 / 2) * s)
    up   = int(h / 2 - target / 2 + (h0 / 2 - ty) * s)
    out = ldms.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] + left) / float(w) * float(w0)
    out[:, 1] = (out[:, 1] + up)   / float(h) * float(h0)
    return out

def unsharp(img_bgr, amount=0.35, radius=1.2):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# =============================== IO Bundle ====================================
def load_bundle(stem_dir: Path):
    mesh_dir = stem_dir / "mesh"
    masks_dir = stem_dir / "masks_uv"
    manifest = json.loads((stem_dir / "manifest.json").read_text(encoding="utf-8"))

    V  = np.load(mesh_dir / "verts.npy").astype(np.float32)
    F  = np.load(mesh_dir / "faces.npy").astype(np.int32)
    UV = np.load(mesh_dir / "uv_coords.npy").astype(np.float32)
    U  = np.load(mesh_dir / "u_mean.npy").astype(np.float32)

    pro = json.loads((mesh_dir / "procrustes.json").read_text(encoding="utf-8"))
    s, R, t = pro["s"], np.asarray(pro["R"], np.float64), np.asarray(pro["t"], np.float64)
    U_aligned = (s * (U.astype(np.float64) @ R.T)) + t
    U_aligned = U_aligned.astype(np.float32)

    masks8 = np.load(masks_dir / "masks_uv_coarse8.npy")  # HxWx8 uint8
    trans_params = np.load(mesh_dir / "trans_params.npy")

    tex_src = Path(manifest.get("source_tex", "")) if "source_tex" in manifest else None
    if not tex_src or not tex_src.exists():
        guess = stem_dir / f"{stem_dir.name}.png"
        tex_src = guess if guess.exists() else None
    source_image = Path(manifest.get("source_image", "")) if "source_image" in manifest else None
    return V, F, UV, U_aligned, masks8, trans_params, tex_src, source_image

def save_obj_with_uv_texture(obj_out_path, verts_xyz, faces, uv_coords01, tex_src_png_path, flip_v=False):
    obj_out_path = Path(obj_out_path); out_dir = obj_out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_dst_png_path = obj_out_path.with_suffix(".png")
    if tex_src_png_path and Path(tex_src_png_path).exists():
        if Path(tex_src_png_path).resolve() != tex_dst_png_path.resolve():
            cv2.imwrite(str(tex_dst_png_path), cv2.imread(str(tex_src_png_path)))
    mtl_out_path = obj_out_path.with_suffix(".mtl")
    tri1 = faces.astype(np.int64).copy() + 1
    vt = uv_coords01[:, :2].copy()
    if flip_v: vt[:, 1] = 1.0 - vt[:, 1]
    with open(obj_out_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_out_path.name}\nusemtl material_0\n")
        for v in verts_xyz: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in vt:        f.write(f"vt {t[0]} {t[1]}\n")
        for a, b, c in tri1: f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    with open(mtl_out_path, "w", encoding="utf-8") as m:
        m.write("newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n")
        m.write(f"map_Kd {tex_dst_png_path.name}\n")

def _rewrite_mtl_to_texture(obj_path, new_png_path):
    mtl_path = Path(obj_path).with_suffix(".mtl")
    if not mtl_path.exists(): return
    lines = mtl_path.read_text(encoding="utf-8").splitlines()
    mtl_path.write_text("\n".join("map_Kd "+Path(new_png_path).name if ln.lower().startswith("map_kd ")
                                  else ln for ln in lines), encoding="utf-8")


# =============================== Pesi & scala =================================
def build_weights_and_masks(V, F, UV, masks8):
    H, W, _ = masks8.shape
    eyes  = np.maximum(masks8[:, :, 0], masks8[:, :, 1])
    brows = np.maximum(masks8[:, :, 2], masks8[:, :, 3])
    lips  = np.maximum(masks8[:, :, 5], masks8[:, :, 6])
    nose  = masks8[:, :, 4]
    skin  = masks8[:, :, 7]

    target_bin = ((eyes | brows | lips | nose) > 0).astype(np.uint8)
    w_local_uv  = feather_mask(target_bin * 255, band_px=28)
    w_global_uv = (skin.astype(np.float32) / 255.0) * 0.15

    W_local_soft = _bilinear_uv_sample(UV, w_local_uv)
    W_global     = _bilinear_uv_sample(UV, w_global_uv)

    M_lips  = _bilinear_uv_sample(UV, (lips.astype(np.float32)  / 255.0))
    M_eyes  = _bilinear_uv_sample(UV, (eyes.astype(np.float32)  / 255.0))
    M_nose  = _bilinear_uv_sample(UV, (nose.astype(np.float32)  / 255.0))
    # smoothing geodetico
    W_local_soft = diffuse_vertex_scalar(V, F, W_local_soft, tau=0.40, iters=12)
    W_global     = diffuse_vertex_scalar(V, F, W_global,     tau=0.35, iters=10)
    M_lips       = diffuse_vertex_scalar(V, F, M_lips,       tau=0.45, iters=10)
    M_eyes       = diffuse_vertex_scalar(V, F, M_eyes,       tau=0.45, iters=10)
    M_nose       = diffuse_vertex_scalar(V, F, M_nose,       tau=0.45, iters=10)
    return W_local_soft, W_global, M_lips, M_eyes, M_nose

def _face_scale(V, UV, masks8):
    right_eye = masks8[:, :, 0].astype(np.float32) / 255.0
    left_eye  = masks8[:, :, 1].astype(np.float32) / 255.0
    skin      = masks8[:, :, 7].astype(np.float32) / 255.0
    MeR = _bilinear_uv_sample(UV, right_eye); MeL = _bilinear_uv_sample(UV, left_eye)
    Ms  = _bilinear_uv_sample(UV, skin)
    idxR = np.where(MeR > 0.5)[0]; idxL = np.where(MeL > 0.5)[0]
    if idxR.size > 10 and idxL.size > 10:
        return float(np.linalg.norm(V[idxR].mean(0) - V[idxL].mean(0)))
    idxS = np.where(Ms > 0.3)[0]
    if idxS.size > 10:
        bbmin = V[idxS].min(0); bbmax = V[idxS].max(0)
        return float(np.linalg.norm(bbmax - bbmin)) * 0.65
    bbmin = V.min(0); bbmax = V.max(0)
    return float(np.linalg.norm(bbmax - bbmin)) * 0.50

def _auto_gamma(V, U_aligned, mask, face_scale, anon_ratio=0.085):
    disp = (U_aligned - V); mags = np.linalg.norm(disp, axis=1)
    cur = robust_percentile(mags[mask > 0.2], 90.0) if np.any(mask > 0.2) else robust_percentile(mags, 90.0)
    desired = max(1e-6, float(anon_ratio) * float(face_scale))
    gamma = desired / max(cur, 1e-8)
    return float(np.clip(gamma, 0.7, 2.4))


# =============================== ARAP solver ==================================
def arap_solve_soft(V, F, target_pos, w_data=45.0, iters=10):
    V0 = V.astype(np.float64).copy()
    n = V.shape[0]
    L = cotmatrix(V0, F)
    if np.isscalar(w_data):
        w_vec = np.full(n, float(w_data), dtype=np.float64)
    else:
        w_vec = np.asarray(w_data, np.float64).reshape(-1)
    A = (L + sp.diags(w_vec)).tocsr()
    d = np.maximum(L.diagonal(), 1e-12)
    W = sp.diags(d) - L
    indptr, indices, data = W.indptr, W.indices, W.data
    neighbors = [indices[indptr[i]:indptr[i+1]] for i in range(n)]
    weights   = [data[indptr[i]:indptr[i+1]]     for i in range(n)]
    X = V0.copy()
    for _ in range(max(1, iters)):
        R = [np.eye(3) for _ in range(n)]
        for i in range(n):
            Pi = np.zeros((3, 3))
            vi0 = V0[i]; xi = X[i]
            for j, wij in zip(neighbors[i], weights[i]):
                pij0 = (V0[j] - vi0).reshape(3, 1)
                pij  = (X[j]  - xi ).reshape(3, 1)
                Pi += float(wij) * (pij @ pij0.T)
            U, _, VT = np.linalg.svd(Pi); Ri = U @ VT
            if np.linalg.det(Ri) < 0: U[:, -1] *= -1; Ri = U @ VT
            R[i] = Ri
        B = (w_vec[:, None]) * target_pos
        for d_ in range(3):
            X[:, d_] = spla.spsolve(A, B[:, d_])
    return X.astype(np.float32)


# =============================== Morph core ===================================
def morph(V, U_aligned, F, UV, masks8, strength=1.0, focus_power=1.8):
    # 1) pesi & maschere
    W_local, W_global, M_lips, M_eyes, M_nose = build_weights_and_masks(V, F, UV, masks8)
    s = smootherstep01(W_local, 0.10, 0.85)
    focus = np.clip(W_local, 0.0, 1.0) ** float(focus_power)
    gscale = 0.12 + 0.43 * focus  # fisso (inside/outside default)

    # 2) intensità adattiva
    face_scale = _face_scale(V, UV, masks8)
    base_ratio = 0.085 * float(strength)    # unico "knob" esterno
    mask_for_gamma = np.clip(W_local + 0.10 * W_global, 0.0, 1.0)
    gamma = _auto_gamma(V, U_aligned, mask_for_gamma, face_scale, anon_ratio=base_ratio)
    disp = (U_aligned - V) * float(gamma)

    # 3) mix locale/globale (gain fissi per eyes/nose, regolabile solo mouth)
    beta_local, beta_global = 0.95, 0.30
    mouth_gain = 1.55
    local_gain = (
        1.0
        + (mouth_gain - 1.0) * M_lips
        + 0.25 * M_eyes      # (= eyes_gain-1)
        + 0.15 * M_nose      # (= nose_gain-1)
    )
    mix = beta_global * (1.0 - s) * np.clip(W_global, 0.0, 1.0) * gscale + beta_local * s * local_gain
    V_target = V + mix[:, None] * disp

    # 4) solve ARAP + smoothing Poisson del displacement
    w_vec = 45.0 * (0.25 + 0.55 * s + 0.20 * np.clip(W_global, 0.0, 1.0))
    V_new = arap_solve_soft(V, F, target_pos=V_target, w_data=w_vec, iters=10)
    alpha_eff = 0.12 * (0.6 + 0.8 * float(np.mean(s * (1.0 - s))))
    V_new = (V + screened_poisson_smooth(V, F, (V_new - V), alpha=alpha_eff)).astype(np.float32)
    return V_new, gamma, face_scale


# =============================== Re-bake UV ===================================
def _get_src_and_trans(stem_dir: Path):
    stem_dir = Path(stem_dir); mesh_dir = stem_dir / "mesh"
    manifest = json.loads((stem_dir / "manifest.json").read_text(encoding="utf-8"))
    src_img_path = manifest.get("source_image", "")
    if not src_img_path or not Path(src_img_path).exists():
        print("  [re-bake] source_image mancante → salto re-bake."); return None, None
    trans_path = mesh_dir / "trans_params.npy"
    if not trans_path.exists():
        print("  [re-bake] trans_params.npy mancante → salto re-bake."); return None, None
    trans_params = np.load(str(trans_path)).astype(np.float32)
    src = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"[re-bake] Immagine non leggibile: {src_img_path}")
    return src, trans_params

def _apply_eye_black_uv(atlas_bgr, stem_dir, feather_px=6, darkness=0):
    masks8_path = Path(stem_dir) / "masks_uv" / "masks_uv_coarse8.npy"
    if not masks8_path.exists(): return atlas_bgr
    masks8 = np.load(str(masks8_path))
    mask_eyes = np.maximum(masks8[:, :, 0], masks8[:, :, 1]).astype(np.float32) / 255.0
    if mask_eyes.max() <= 0.0: return atlas_bgr
    H, W = atlas_bgr.shape[:2]
    m = cv2.resize(mask_eyes, (W, H), interpolation=cv2.INTER_LINEAR)
    m = cv2.flip(m, 0)
    if feather_px and feather_px > 0:
        k = max(1, int(feather_px))
        m = cv2.GaussianBlur(m, (0, 0), k * 0.7)
    m3 = m[:, :, None].astype(np.float32)
    base = np.full_like(atlas_bgr, darkness, dtype=np.uint8)
    out = (base.astype(np.float32) * m3 + atlas_bgr.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
    return out

def rebake_uv_png_per_triangle(stem_dir, V_for_proj, F, UV, out_png_path,
                               uv_res=4096, downsample=2, do_sharpen=True,
                               feather_px=2, min_uv_area_px=3.0,
                               eye_black=False, eye_dark_value=0):
    src, trans_params = _get_src_and_trans(stem_dir)
    if src is None: return None
    R = int(max(512, uv_res))
    atlas = np.zeros((R, R, 3), np.uint8)
    uv_px = np.column_stack([UV[:, 0]*(R-1), (1.0-UV[:, 1])*(R-1)]).astype(np.float32)
    v2d = _project_to_crop224(V_for_proj).astype(np.float32); v2d[:, 1] = 223.0 - v2d[:, 1]
    v2d_full = _back_resize_ldms(v2d, trans_params).astype(np.float32)

    tri_int = F.astype(np.int64)
    for (a, b, c) in tri_int:
        dst = np.float32([uv_px[a], uv_px[b], uv_px[c]])
        area = 0.5*abs((dst[1,0]-dst[0,0])*(dst[2,1]-dst[0,1]) - (dst[2,0]-dst[0,0])*(dst[1,1]-dst[0,1]))
        if area < float(min_uv_area_px): continue
        src_tri = np.float32([v2d_full[a], v2d_full[b], v2d_full[c]])
        x0 = int(np.floor(np.min(dst[:,0])));  x1 = int(np.ceil(np.max(dst[:,0]))) + 1
        y0 = int(np.floor(np.min(dst[:,1])));  y1 = int(np.ceil(np.max(dst[:,1]))) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= R or y0 >= R: continue
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(R, x1); y1 = min(R, y1)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1: continue
        dst_roi = dst.copy(); dst_roi[:, 0] -= x0; dst_roi[:, 1] -= y0
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_roi + 0.5), 255, lineType=cv2.LINE_AA)
        if feather_px > 0:
            k = max(1, int(feather_px * max(1.0, (area ** 0.5) / 12.0)))
            mask = cv2.GaussianBlur(mask, (0, 0), k * 0.6)
        M = cv2.getAffineTransform(src_tri, dst_roi)  # src → dst (ROI)
        patch = cv2.warpAffine(src, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m3 = (mask.astype(np.float32)/255.0)[:, :, None]
        atlas[y0:y1, x0:x1] = (patch*m3 + atlas[y0:y1, x0:x1]*(1.0-m3)).astype(np.uint8)

    outR = R // max(1, int(downsample))
    atlas_ds = cv2.resize(atlas, (outR, outR), interpolation=cv2.INTER_AREA)
    if eye_black:
        atlas_ds = _apply_eye_black_uv(atlas_ds, stem_dir, feather_px=6, darkness=int(eye_dark_value))
    if do_sharpen:
        atlas_ds = unsharp(atlas_ds, amount=0.35, radius=1.2)
    out_png_path = Path(out_png_path); out_png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png_path), atlas_ds, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"  [re-bake] PNG: {out_png_path}")
    return str(out_png_path)


# =============================== Pipeline =====================================
def process_all(bundles_dir="stage3_bundles", out_dir="stage3_morphed_slim",
                strength=1.0, focus=1.8,
                rebake_res=4096, sharpen=1, eye_black=1, eye_dark_value=0, flip_v=0):
    bundles_dir = Path(bundles_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stems = [p for p in bundles_dir.iterdir() if p.is_dir()]
    if not stems:
        print(f"Nessun bundle in {bundles_dir}"); return
    for stem_dir in stems:
        stem = stem_dir.name
        print(f"[{stem}] → carico bundle…")
        V, F, UV, U_aligned, masks8, trans_params, tex_src, source_image = load_bundle(stem_dir)
        V_new, gamma, face_scale = morph(V, U_aligned, F, UV, masks8, strength=float(strength), focus_power=float(focus))
        print(f"  morph: gamma={gamma:.3f} | face_scale={face_scale:.3f}")

        # salva mesh + OBJ con texture originale
        out_mesh_dir = out_dir / stem / "mesh"; out_mesh_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_mesh_dir / "verts_morphed.npy", V_new.astype(np.float32))
        out_obj = out_dir / stem / f"{stem}_morphed.obj"
        save_obj_with_uv_texture(out_obj, V_new, F, UV, tex_src, flip_v=bool(flip_v))
        print(f"  OBJ salvato: {out_obj}")

        # re-bake UV — usa SEMPRE la mesh ORIGINALE per il campionamento dalla foto
        rebaked_png = out_obj.with_name(out_obj.stem + "_rebaked.png")
        png_path = rebake_uv_png_per_triangle(
            stem_dir, V, F, UV, rebaked_png,
            uv_res=int(rebake_res), downsample=2, do_sharpen=bool(sharpen),
            eye_black=bool(eye_black), eye_dark_value=int(eye_dark_value)
        )
        if png_path:
            _rewrite_mtl_to_texture(out_obj, png_path)
            print(f"  MTL aggiornato → {Path(png_path).name}")

def main():
    ap = argparse.ArgumentParser(description="Fase 3 — SLIM (pochi parametri, qualità invariata)")
    ap.add_argument("--bundles_dir", default="stage3_bundles")
    ap.add_argument("--out_dir", default="stage3_morphed_slim")
    ap.add_argument("--strength", type=float, default=1.0, help="0.5=più soft, 1.0=default, 1.5=più forte")
    ap.add_argument("--focus", type=float, default=1.8, help=">1 concentra su labbra/occhi/naso")
    ap.add_argument("--rebake_res", type=int, default=4096)
    ap.add_argument("--sharpen", type=int, default=1)
    ap.add_argument("--eye_black", type=int, default=1)
    ap.add_argument("--eye_dark_value", type=int, default=0)
    ap.add_argument("--flip_v", type=int, default=0)
    args = ap.parse_args()
    process_all(
        bundles_dir=args.bundles_dir, out_dir=args.out_dir,
        strength=args.strength, focus=args.focus,
        rebake_res=args.rebake_res, sharpen=args.sharpen,
        eye_black=args.eye_black, eye_dark_value=args.eye_dark_value, flip_v=args.flip_v
    )

if __name__ == "__main__":
    main()
