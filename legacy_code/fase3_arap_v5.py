import json
import shutil
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Utils numerici
# =============================================================================
def smootherstep01(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Quintic smootherstep: C2-continuo."""
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0).astype(np.float32)
    return t*t*t*(t*(t*6.0 - 15.0) + 10.0)

def robust_percentile(x, p=95.0):
    x = np.asarray(x, np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    p = np.clip(float(p), 0.0, 100.0)
    return float(np.percentile(np.abs(x), p))


# =============================================================================
# I/O: OBJ + MTL + PNG
# =============================================================================
def save_obj_with_uv_texture(obj_out_path, verts_xyz, faces, uv_coords01, tex_src_png_path, flip_v=False):
    """
    Scrive OBJ+MTL e copia la PNG accanto all'OBJ.
    - verts_xyz: (N,3) float32
    - faces: (F,3) int32 0-based
    - uv_coords01: (N,2) float32 [0,1]
    - tex_src_png_path: file png da copiare accanto
    """
    obj_out_path = Path(obj_out_path)
    out_dir = obj_out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_dst_png_path = obj_out_path.with_suffix(".png")
    tex_src_png_path = Path(tex_src_png_path)
    if not tex_src_png_path.exists():
        raise FileNotFoundError(f"Texture PNG non trovata: {tex_src_png_path}")
    if tex_src_png_path.resolve() != tex_dst_png_path.resolve():
        shutil.copyfile(tex_src_png_path, tex_dst_png_path)

    mtl_out_path = obj_out_path.with_suffix(".mtl")
    tri1 = faces.astype(np.int64).copy() + 1  # OBJ 1-based

    vt = uv_coords01[:, :2].copy()
    if flip_v:
        vt[:, 1] = 1.0 - vt[:, 1]

    with open(obj_out_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_out_path.name}\n")
        f.write("usemtl material_0\n")
        for v in verts_xyz:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in vt:
            f.write(f"vt {t[0]} {t[1]}\n")
        for a, b, c in tri1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    with open(mtl_out_path, "w", encoding="utf-8") as m:
        m.write("newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n")
        m.write(f"map_Kd {tex_dst_png_path.name}\n")


def _rewrite_mtl_to_texture(obj_path, new_png_path):
    mtl_path = Path(obj_path).with_suffix(".mtl")
    if not mtl_path.exists():
        return
    lines = mtl_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for ln in lines:
        if ln.strip().lower().startswith("map_kd "):
            new_lines.append(f"map_Kd {Path(new_png_path).name}")
        else:
            new_lines.append(ln)
    mtl_path.write_text("\n".join(new_lines), encoding="utf-8")


# =============================================================================
# LOAD BUNDLE
# =============================================================================
def load_bundle(stem_dir: Path):
    mesh_dir = stem_dir / "mesh"
    masks_dir = stem_dir / "masks_uv"
    manifest_path = stem_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json mancante in {stem_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    V = np.load(mesh_dir / "verts.npy")        # (N,3)
    F = np.load(mesh_dir / "faces.npy")        # (F,3)
    UV = np.load(mesh_dir / "uv_coords.npy")   # (N,2) [0,1]
    U = np.load(mesh_dir / "u_mean.npy")       # (N,3)

    tex_src = Path(manifest.get("source_tex", "")) if "source_tex" in manifest else None
    if not tex_src or not tex_src.exists():
        guess1 = stem_dir / f"{stem_dir.name}.png"
        tex_src = guess1 if guess1.exists() else None

    # procrustes per allineare U alla posa del soggetto
    pro_path = mesh_dir / "procrustes.json"
    if not pro_path.exists():
        raise FileNotFoundError(f"procrustes.json mancante in {mesh_dir}")
    pro = json.loads(pro_path.read_text(encoding="utf-8"))
    s, R, t = pro["s"], np.asarray(pro["R"], np.float64), np.asarray(pro["t"], np.float64)
    U_aligned = (s * (U.astype(np.float64) @ R.T)) + t
    U_aligned = U_aligned.astype(np.float32)

    masks8_path = masks_dir / "masks_uv_coarse8.npy"
    if not masks8_path.exists():
        raise FileNotFoundError(f"masks_uv_coarse8.npy mancante in {masks_dir}")
    masks8 = np.load(masks8_path)  # HxWx8, uint8

    trans_path = mesh_dir / "trans_params.npy"
    trans_params = np.load(trans_path) if trans_path.exists() else None
    source_image = Path(manifest.get("source_image", "")) if "source_image" in manifest else None

    return V, F, UV, U_aligned, tex_src, manifest, masks8, trans_params, source_image


# =============================================================================
# UV SOFT-WEIGHTS + per-vertex masks
# =============================================================================
def feather_mask(mask_u8, band_px):
    """
    Piumatura continua basata su Signed Distance Field.
    - Restituisce un peso in [0,1] con pendenza morbida e continua (C2 approx),
      evitando il "gradino" interno che produce stacchi visivi.
    """
    m = (mask_u8.astype(np.uint8) > 0).astype(np.uint8)
    if band_px <= 0:
        return m.astype(np.float32)

    # Distanze inside/outside
    dist_out = cv2.distanceTransform((1 - m) * 255, cv2.DIST_L2, 3).astype(np.float32)
    dist_in  = cv2.distanceTransform(m * 255,     cv2.DIST_L2, 3).astype(np.float32)
    sdf = dist_out - dist_in  # >0 fuori, <0 dentro

    # Mappa SDF → [0,1] con transizione morbida nell'intorno di 0
    k = max(1.0, float(band_px) * 0.75)
    w = 0.5 * (1.0 - np.tanh(sdf / k)).astype(np.float32)
    return np.clip(w, 0.0, 1.0)


def _bilinear_uv_sample(UV: np.ndarray, img01: np.ndarray) -> np.ndarray:
    """Campiona img01 (H,W) float32 in posizioni UV per-vertice → (N,)"""
    H, W = img01.shape[:2]
    u = np.clip(UV[:, 0] * (W - 1), 0, W - 1)
    v = np.clip(UV[:, 1] * (H - 1), 0, H - 1)
    x0 = np.floor(u).astype(np.int32); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(v).astype(np.int32); y1 = np.clip(y0 + 1, 0, H - 1)
    Ia = img01[y0, x0]
    Ib = img01[y1, x0]
    Ic = img01[y0, x1]
    Id = img01[y1, x1]
    wa = (x1 - u) * (y1 - v)
    wb = (x1 - u) * (v - y0)
    wc = (u - x0) * (y1 - v)
    wd = (u - x0) * (v - y0)
    return np.clip(wa * Ia + wb * Ib + wc * Ic + wd * Id, 0.0, 1.0).astype(np.float32)

def diffuse_vertex_scalar(V, F, w0, tau=0.40, iters=12):
    """
    Diffusione su mesh (geodetica) dei pesi per eliminare aliasing UV.
    w <- (1 - tau) w + tau * media_pesata_vicini  (con pesi cotangenti normalizzati).
    """
    L = cotmatrix(V, F)                    # L = D - W
    d = np.maximum(L.diagonal(), 1e-12)
    W = sp.diags(d) - L
    w = np.asarray(w0, np.float64).reshape(-1)
    for _ in range(max(1, int(iters))):
        neigh = W.dot(w) / d
        w = (1.0 - tau) * w + tau * neigh
    return np.clip(w, 0.0, 1.0).astype(np.float32)

def build_weights_and_masks(UV, masks8, band_px=32, skin_floor=0.12):
    """
    Pesi UV con SDF + diffusione su mesh applicata dopo il campionamento per-vertice.
    """
    mask_eyes  = np.maximum(masks8[:, :, 0], masks8[:, :, 1])
    mask_brows = np.maximum(masks8[:, :, 2], masks8[:, :, 3])
    mask_lips  = np.maximum(masks8[:, :, 5], masks8[:, :, 6])
    mask_nose  = masks8[:, :, 4]
    mask_skin  = masks8[:, :, 7]

    # SDF-feather sulle feature locali
    target_bin = ((mask_eyes | mask_brows | mask_lips | mask_nose) > 0).astype(np.uint8)
    w_local_uv  = feather_mask(target_bin * 255, band_px)                # [0,1] morbido
    w_global_uv = (mask_skin.astype(np.float32) / 255.0) * float(skin_floor)

    # Campionamento UV → per-vertex
    W_local_soft = _bilinear_uv_sample(UV, w_local_uv)
    W_global     = _bilinear_uv_sample(UV, w_global_uv)

    M_eyes  = _bilinear_uv_sample(UV, (mask_eyes.astype(np.float32)  / 255.0))
    M_lips  = _bilinear_uv_sample(UV, (mask_lips.astype(np.float32)  / 255.0))
    M_skin  = _bilinear_uv_sample(UV, (mask_skin.astype(np.float32)  / 255.0))
    M_brows = _bilinear_uv_sample(UV, (mask_brows.astype(np.float32) / 255.0))
    M_nose  = _bilinear_uv_sample(UV, (mask_nose.astype(np.float32)  / 255.0))

    # Diffusione su mesh per eliminare aliasing UV e rendere geodetico il bordo
    # (importante per giunzioni bocca/guancia e naso/fronte)
    # Nota: qui non abbiamo V,F – li diffondiamo in morph_fused_solver quando disponibili.
    return W_local_soft, W_global, M_eyes, M_lips, M_skin, M_brows, M_nose



# =============================================================================
# LAPLACIANO COTANGENTE + MASS MATRIX
# =============================================================================
def cotmatrix(V, F):
    """
    Laplaciano cotangente simmetrico L (csr, NxN). V: (N,3), F: (M,3)
    """
    V = V.astype(np.float64)
    F = F.astype(np.int64)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    e0 = v1 - v2
    e1 = v2 - v0
    e2 = v0 - v1

    def _cot(a, b):
        num = (a * b).sum(axis=1)
        cross = np.cross(a, b)
        den = np.linalg.norm(cross, axis=1) + 1e-12
        return num / den

    cot0 = _cot(e1, e2)  # opposto a v0
    cot1 = _cot(e2, e0)  # opposto a v1
    cot2 = _cot(e0, e1)  # opposto a v2

    I = np.concatenate([i1, i2, i0, i2, i0, i1])
    J = np.concatenate([i2, i1, i2, i0, i1, i0])
    W = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    n = V.shape[0]
    W = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    d = np.array(W.sum(axis=1)).ravel()
    L = sp.diags(d) - W
    return L


def mass_matrix_lumped(V, F):
    Vd = V.astype(np.float64); F = F.astype(np.int64)
    v0, v1, v2 = Vd[F[:, 0]], Vd[F[:, 1]], Vd[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # area tri
    n = V.shape[0]
    M = sp.lil_matrix((n, n))
    for k, (a, b, c) in enumerate(F):
        w = A[k] / 3.0
        M[a, a] += w; M[b, b] += w; M[c, c] += w
    return M.tocsr()


# =============================================================================
# SOLVERS: armonico, bi-armonico, ARAP (soft per-vertice)
# =============================================================================
def harmonic_interpolate(V, F, fixed_idx, fixed_pos):
    """
    L * X = 0 (cot-Laplacian) con vincoli Dirichlet su fixed_idx. Ritorna X (N,3)
    """
    n = V.shape[0]
    L = cotmatrix(V, F)
    fixed = np.zeros(n, dtype=bool); fixed[fixed_idx] = True
    free_idx = np.where(~fixed)[0]
    if free_idx.size == 0:
        X = V.copy(); X[fixed_idx] = fixed_pos; return X
    L_ff = L[free_idx][:, free_idx]
    L_fc = L[free_idx][:, fixed_idx]
    B = - L_fc.dot(fixed_pos)
    X = V.copy().astype(np.float64)
    for d in range(3):
        X[free_idx, d] = spla.spsolve(L_ff, B[:, d])
    X[fixed_idx] = fixed_pos
    return X.astype(np.float32)


def biharmonic_interpolate(V, F, fixed_idx, fixed_pos):
    """
    (L^T M^{-1} L) X = 0 con vincoli Dirichlet su fixed_idx. Più morbido (meno pieghe).
    """
    n = V.shape[0]
    L = cotmatrix(V, F)
    M = mass_matrix_lumped(V, F)
    Minv = sp.diags(1.0 / (M.diagonal() + 1e-12))
    A = (L.T @ Minv @ L)  # SPD

    fixed = np.zeros(n, dtype=bool); fixed[fixed_idx] = True
    free = np.where(~fixed)[0]
    if free.size == 0:
        X = V.copy(); X[fixed_idx] = fixed_pos; return X

    A_ff = A[free][:, free]
    A_fc = A[free][:, fixed_idx]
    B = - A_fc.dot(fixed_pos)
    X = V.copy().astype(np.float64)
    for d in range(3):
        X[free, d] = spla.cg(A_ff, B[:, d], tol=1e-6, maxiter=600)[0]
    X[fixed_idx] = fixed_pos
    return X.astype(np.float32)

def screened_poisson_smooth(V, F, disp, alpha=0.12):
    """
    Risolve (M - alpha * L) d_smooth = M * d   (SPD).
    alpha piccolo = poco smoothing; grande = più liscio.
    """
    L = cotmatrix(V, F)
    M = mass_matrix_lumped(V, F)
    A = (M - float(alpha) * L).tocsr()
    B = M.dot(disp.astype(np.float64))
    X = np.empty_like(disp, dtype=np.float64)
    for k in range(3):
        X[:, k] = spla.spsolve(A, B[:, k])
    return X.astype(np.float32)

def arap_solve_soft(V, F, target_pos, w_data=50.0, iters=6):
    """
    ARAP local-global con vincoli soft. Rotazioni pesate da cotangenti (più stabili sui bordi).
    """
    V0 = V.astype(np.float64).copy()
    n = V.shape[0]
    L = cotmatrix(V0, F)

    # w per-vertice
    if np.isscalar(w_data):
        w_vec = np.full(n, float(w_data), dtype=np.float64)
    else:
        w_vec = np.asarray(w_data, dtype=np.float64).reshape(-1)
        if w_vec.size != n:
            raise ValueError("w_data deve avere dimensione N")

    # Sistema globale
    A = (L + sp.diags(w_vec)).tocsr()

    # Pesi di adiacenza (cotangenti)
    d = np.maximum(L.diagonal(), 1e-12)
    W = sp.diags(d) - L
    indptr, indices, data = W.indptr, W.indices, W.data

    # Liste vicini per ciclo locale
    neighbors = [indices[indptr[i]:indptr[i+1]] for i in range(n)]
    weights   = [data[indptr[i]:indptr[i+1]]     for i in range(n)]

    X = V0.copy()
    for _ in range(max(1, iters)):
        # Local: stima R_i con pesi cotangenti
        R = [np.eye(3) for _ in range(n)]
        for i in range(n):
            Pi = np.zeros((3, 3))
            vi0 = V0[i]
            xi  = X[i]
            for j, wij in zip(neighbors[i], weights[i]):
                pij0 = (V0[j] - vi0).reshape(3, 1)
                pij  = (X[j]  - xi ).reshape(3, 1)
                Pi += float(wij) * (pij @ pij0.T)
            U, _, VT = np.linalg.svd(Pi)
            Ri = U @ VT
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ VT
            R[i] = Ri

        # Global: (L + diag(w)) X = diag(w) * target
        B = (w_vec[:, None]) * target_pos
        for d_ in range(3):
            X[:, d_] = spla.spsolve(A, B[:, d_])
    return X.astype(np.float32)



# =============================================================================
# SUPPORT: adiacenza, proiezioni 2D, sampling
# =============================================================================
def _project_to_crop224(V_obj):
    V_cam = V_obj.astype(np.float32).copy()
    V_cam[:, 2] = 10.0 - V_cam[:, 2]
    f, cx, cy = 1015.0, 112.0, 112.0
    P = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    proj = V_cam @ P.T
    v2d = proj[:, :2] / proj[:, 2:3]
    return v2d  # (N,2) su [0,224)


def _back_resize_ldms(ldms, trans_params):
    w0, h0, s, tx, ty = trans_params
    target_size = 224.0
    w = int(w0 * s); h = int(h0 * s)
    left = int(w / 2 - target_size / 2 + (tx - w0 / 2) * s)
    up   = int(h / 2 - target_size / 2 + (h0 / 2 - ty) * s)
    out = ldms.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] + left) / float(w) * float(w0)
    out[:, 1] = (out[:, 1] + up)   / float(h) * float(h0)
    return out


def _bilinear_sample_rgb(img_bgr, xs, ys):
    img = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0  # RGB [0,1]
    H, W = img.shape[:2]
    x = np.clip(xs, 0, W - 1); y = np.clip(ys, 0, H - 1)
    x0 = np.floor(x).astype(int); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(int); y1 = np.clip(y0 + 1, 0, H - 1)
    Ia = img[y0, x0]; Ib = img[y1, x0]; Ic = img[y0, x1]; Id = img[y1, x1]
    wa = (x1 - x) * (y1 - y); wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y); wd = (x - x0) * (y - y0)
    return (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None]).astype(np.float32)


# =============================================================================
# SCALA VOLTO + INTENSITÀ ADATTIVA
# =============================================================================
def _face_scale_from_masks(V, UV, masks8) -> float:
    """Stima la scala del volto (inter-oculare se possibile, altrimenti bbox skin)."""
    H, W, _ = masks8.shape
    right_eye = masks8[:, :, 0].astype(np.float32) / 255.0
    left_eye  = masks8[:, :, 1].astype(np.float32) / 255.0
    skin      = masks8[:, :, 7].astype(np.float32) / 255.0

    MeR = _bilinear_uv_sample(UV, right_eye)
    MeL = _bilinear_uv_sample(UV, left_eye)
    Ms  = _bilinear_uv_sample(UV, skin)

    idxR = np.where(MeR > 0.5)[0]; idxL = np.where(MeL > 0.5)[0]
    if idxR.size > 10 and idxL.size > 10:
        cR = V[idxR].mean(axis=0)
        cL = V[idxL].mean(axis=0)
        dist = float(np.linalg.norm(cR - cL))
        if dist > 1e-6:
            return dist

    # fallback: diagonale bbox della pelle
    idxS = np.where(Ms > 0.3)[0]
    if idxS.size > 10:
        bbmin = V[idxS].min(axis=0); bbmax = V[idxS].max(axis=0)
        return float(np.linalg.norm(bbmax - bbmin)) * 0.65
    # ultimo fallback: bbox globale
    bbmin = V.min(axis=0); bbmax = V.max(axis=0)
    return float(np.linalg.norm(bbmax - bbmin)) * 0.50


def _auto_strength_factor(V, U_aligned, target_mask, face_scale, anon_ratio=0.08, clip=(0.6, 2.6)):
    """Calcola un gamma tale che la media-robusta degli spostamenti nelle zone target valga anon_ratio*face_scale."""
    disp = (U_aligned - V)
    mags = np.linalg.norm(disp, axis=1)
    cur = robust_percentile(mags[target_mask > 0.2], p=90.0) if np.any(target_mask > 0.2) else robust_percentile(mags, 90.0)
    desired = max(1e-6, float(anon_ratio) * float(face_scale))
    gamma = desired / max(cur, 1e-8)
    return float(np.clip(gamma, clip[0], clip[1]))


# =============================================================================
# MORPH con fusione morbida e pesi per-vertice
# =============================================================================
def morph_fused_solver(V, U_aligned, F, UV, masks8,
                       t_low=0.10, t_high=0.85,
                       beta_local=0.9, beta_global=0.35,
                       mouth_gain=1.5, eyes_gain=1.25, nose_gain=1.15, brows_gain=1.0,
                       focus_power=1.8, global_inside=0.55, global_outside=0.12,
                       solver="arap", arap_w_base=45.0, arap_iters=8,
                       smooth_disp_alpha=0.12,
                       auto_strength=True, anon_ratio=0.085):
    """
    Crea un target continuo (senza stacchi) e risolve con solver soft (ARAP per-vertice consigliato).
    - auto_strength: adatta la potenza su mesh di scala diversa per ottenere ~anon_ratio*face_scale
    - mouth_gain: moltiplicatore del contributo locale sui lips
    """
    # pesi + maschere per-vertice
    W_local_soft, W_global, M_eyes, M_lips, M_skin, M_brows, M_nose = build_weights_and_masks(UV, masks8, band_px=28, skin_floor=0.15)
    M_lips       = diffuse_vertex_scalar(V, F, M_lips,       tau=0.45, iters=10)
    M_eyes       = diffuse_vertex_scalar(V, F, M_eyes,       tau=0.45, iters=10)
    M_brows      = diffuse_vertex_scalar(V, F, M_brows,      tau=0.45, iters=10)
    M_nose       = diffuse_vertex_scalar(V, F, M_nose,       tau=0.45, iters=10)
    # scala volto e gamma adattivo
    face_scale = _face_scale_from_masks(V, UV, masks8)
    gamma = 1.0
    if auto_strength:
        # target mask: features locali (occhi/sopracc/schiena naso/labbra)
        target_mask = np.clip(W_local_soft + 0.10 * W_global, 0.0, 1.0)
        gamma = _auto_strength_factor(V, U_aligned, target_mask, face_scale, anon_ratio=anon_ratio, clip=(0.7, 2.4))

    disp = (U_aligned - V) * float(gamma)

    # gain per-region: lips / eyes / nose / brows
    local_gain = (1.0+ (float(mouth_gain) - 1.0) * np.clip(M_lips,  0.0, 1.0)+ (float(eyes_gain)  - 1.0) * np.clip(M_eyes,  0.0, 1.0)+ (float(nose_gain)  - 1.0) * np.clip(M_nose,  0.0, 1.0)+ (float(brows_gain) - 1.0) * np.clip(M_brows, 0.0, 1.0))

    s = smootherstep01(W_local_soft, t_low, t_high)
    # fattore di focus (0..1) stringendo la regione locale con una potenza
    focus = np.clip(W_local_soft, 0.0, 1.0) ** float(focus_power)
    # pesatura globale: alto "dentro", basso "fuori"
    gscale = float(global_outside) + (float(global_inside) - float(global_outside)) * focus
    # mix complessivo
    mix = (beta_global * (1.0 - s) * np.clip(W_global, 0.0, 1.0) * gscale + beta_local * s * local_gain)

    V_target = V + mix[:, None] * disp

    # pesi soft per-vertice per ARAP/biharmonic: più forte sul locale, medio sul globale
    w_vec = arap_w_base * (0.25 + 0.55 * s + 0.20 * np.clip(W_global, 0.0, 1.0))

    # solve
    solver = str(solver).lower()
    if solver == "biharmonic":
        # usa un set di punti fissi molto sparso (bordo pelle) come ancoraggio, il resto soft via ARAP polish
        idx_anchor = np.where((W_global > 0.12) & (s < 0.05))[0]
        if idx_anchor.size < 6:
            # fallback: ancora qualche punto random sparso
            idx_anchor = np.linspace(0, V.shape[0] - 1, num=min(64, V.shape[0]), dtype=np.int64)
        V_coarse = biharmonic_interpolate(V, F, idx_anchor, V_target[idx_anchor])
        V_new = arap_solve_soft(V_coarse, F, target_pos=V_target, w_data=w_vec, iters=max(4, arap_iters//2))
    elif solver == "harmonic":
        idx_anchor = np.where((W_global > 0.12) & (s < 0.05))[0]
        if idx_anchor.size < 6:
            idx_anchor = np.linspace(0, V.shape[0] - 1, num=min(64, V.shape[0]), dtype=np.int64)
        V_coarse = harmonic_interpolate(V, F, idx_anchor, V_target[idx_anchor])
        V_new = arap_solve_soft(V_coarse, F, target_pos=V_target, w_data=w_vec, iters=max(4, arap_iters//2))
    else:
        # ARAP puro con vincoli soft per-vertice: ottimo per transizioni lisce
        V_new = arap_solve_soft(V, F, target_pos=V_target, w_data=w_vec, iters=arap_iters)

    # smoothing leggero del campo di spostamento, più intenso nella fascia di transizione
    if smooth_disp_alpha and smooth_disp_alpha > 0:
        alpha_eff = float(smooth_disp_alpha) * (0.6 + 0.8 * float(np.mean(s * (1.0 - s))))
        V_new = (V + screened_poisson_smooth(V, F, (V_new - V), alpha=alpha_eff)).astype(np.float32)

    return V_new, gamma, face_scale


# =============================================================================
# RE-BAKE TEXTURE UV (con oscuramento occhi opzionale)
# =============================================================================
def unsharp(img_bgr, amount=0.35, radius=1.2):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _load_masks8(stem_dir: Path):
    masks_path = Path(stem_dir) / "masks_uv" / "masks_uv_coarse8.npy"
    if masks_path.exists():
        return np.load(str(masks_path))
    return None


def _apply_eye_black_uv(atlas_bgr, stem_dir, feather_px=4):
    """
    Oscura la regione degli occhi nell'atlas UV.
    - darkness: valore di intensità (0=nero, 20=molto scuro ma non nero)
    Nota: l'atlas è già in orientamento con v-flip applicato; flippo anche la maschera.
    """
    masks8 = _load_masks8(stem_dir)
    if masks8 is None:
        return atlas_bgr
    mask_eyes = np.maximum(masks8[:, :, 0], masks8[:, :, 1]).astype(np.float32) / 255.0
    if mask_eyes.max() <= 0.0:
        return atlas_bgr
    # flip verticale per allineare y
    m = cv2.flip(mask_eyes, 0)
    # ammorbidisci bordi
    if feather_px and feather_px > 0:
        k = max(1, int(feather_px))
        m = cv2.GaussianBlur(m, (0, 0), k * 0.7)
    m3 = np.clip(m[:, :, None], 0.0, 1.0)
    # fondi verso nero (o un valore scuro)
    base = np.full_like(atlas_bgr, fill_value=0, dtype=np.uint8)
    out = (base.astype(np.float32) * m3 + atlas_bgr.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
    return out


def rebake_uv_png_per_triangle(stem_dir, V_proj, F, UV, out_png_path,
                               uv_res=4096, do_sharpen=True,
                               feather_px=1, min_uv_area_px=3.0, threads=0,
                               eye_black=False):
    """
    Re-bake veloce: warp affine SOLO sulla ROI (bbox del triangolo in UV) + skip
    dei triangoli con area UV troppo piccola. Molto più rapido di warp su R×R.
    Con eye_black=True oscura la regione degli occhi nell'atlas finale.
    """
    import json
    stem_dir = Path(stem_dir)
    mesh_dir = stem_dir / "mesh"
    manifest = json.loads((stem_dir / "manifest.json").read_text(encoding="utf-8"))

    # threading OpenCV (0=auto, >0 forza il numero di thread)
    try:
        if threads and threads > 0:
            cv2.setNumThreads(int(threads))
    except Exception:
        pass

    src_img_path = manifest.get("source_image", "")
    if not src_img_path or not Path(src_img_path).exists():
        print("  [re-bake] source_image mancante → salto re-bake.")
        return None

    trans_path = mesh_dir / "trans_params.npy"
    if not trans_path.exists():
        print("  [re-bake] trans_params.npy mancante → salto re-bake.")
        return None
    trans_params = np.load(str(trans_path)).astype(np.float32)

    R = int(max(512, uv_res))
    atlas = np.zeros((R, R, 3), np.uint8)
    uv_px = np.column_stack([UV[:, 0]*(R-1), (1.0-UV[:, 1])*(R-1)]).astype(np.float32)

    # proiezione 3D->2D originale
    v2d_crop = _project_to_crop224(V_proj).astype(np.float32)
    v2d_crop[:, 1] = 223.0 - v2d_crop[:, 1]
    v2d_full = _back_resize_ldms(v2d_crop, trans_params).astype(np.float32)

    src = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"[re-bake] Immagine non leggibile: {src_img_path}")

    tri_int = F.astype(np.int64)
    min_area = float(min_uv_area_px)

    for (a, b, c) in tri_int:
        dst = np.float32([uv_px[a], uv_px[b], uv_px[c]])
        # skip triangoli minuscoli in UV
        area = 0.5*abs(
            (dst[1,0]-dst[0,0])*(dst[2,1]-dst[0,1]) - (dst[2,0]-dst[0,0])*(dst[1,1]-dst[0,1])
        )
        if area < min_area:
            continue

        src_tri = np.float32([v2d_full[a], v2d_full[b], v2d_full[c]])
        # bbox ROI nel target
        x0 = int(np.floor(np.min(dst[:,0])));  x1 = int(np.ceil(np.max(dst[:,0]))) + 1
        y0 = int(np.floor(np.min(dst[:,1])));  y1 = int(np.ceil(np.max(dst[:,1]))) + 1
        # clip ai bordi dell'atlas
        if x1 <= 0 or y1 <= 0 or x0 >= R or y0 >= R:
            continue
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(R, x1); y1 = min(R, y1)
        bw = x1 - x0; bh = y1 - y0
        if bw <= 1 or bh <= 1:
            continue

        # triangolo in coord ROI (traslato)
        dst_roi = dst.copy()
        dst_roi[:, 0] -= x0
        dst_roi[:, 1] -= y0

        # maschera ROI
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_roi + 0.5), 255)
        if feather_px > 0:
            # anti-alias proporzionale all'area (evita contorni seghettati visibili)
            k = max(1, int(feather_px * max(1.0, (area ** 0.5) / 12.0)))
            mask = cv2.GaussianBlur(mask, (0, 0), k * 0.6)

        # matrice affine src->ROI: sottraggo l'offset della ROI sul termine di traslazione
        M = cv2.getAffineTransform(src_tri, dst_roi)
        patch = cv2.warpAffine(src, M, (bw, bh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

        # compositing nella ROI
        m3 = (mask.astype(np.float32)/255.0)[:, :, None]
        atlas[y0:y1, x0:x1] = (patch*m3 + atlas[y0:y1, x0:x1]*(1.0-m3)).astype(np.uint8)
    atlas_out = atlas
    if eye_black:
        atlas_out = _apply_eye_black_uv(atlas_out, stem_dir, feather_px=6)
    if do_sharpen:
        atlas_out = unsharp(atlas_out, amount=0.35, radius=1.2)

    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png_path), atlas_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"  [re-bake] PNG (ROI fast): {out_png_path}")
    return str(out_png_path)


def rebake_uv_png(stem_dir, V_proj, F, UV, out_png_path, uv_res, sharpen=1, eye_black=False):
    return rebake_uv_png_per_triangle(
        stem_dir, V_proj, F, UV, out_png_path,
        uv_res=max(1024, uv_res), do_sharpen=bool(sharpen),
        eye_black=bool(eye_black)
    )


# =============================================================================
# PIPELINE
# =============================================================================
def process_all(bundles_dir, out_dir,beta_local, beta_global,t_low, t_high,mouth_gain, eyes_gain, nose_gain, brows_gain,focus_power, 
                global_inside,global_outside,solver, arap_w, arap_iters,smooth_disp_alpha,auto_strength, anon_ratio,
                # re-bake
                rebake, rebake_uv_res, sharpen,eye_black, flip_v=False):

    bundles_dir = Path(bundles_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = [p for p in bundles_dir.iterdir() if p.is_dir()]
    if not stems:
        print(f"Nessun bundle trovato in {bundles_dir}")
        return

    for stem_dir in stems:
        stem = stem_dir.name
        print(f"[{stem}] Carico bundle…")
        V, F, UV, U_aligned, tex_src, manifest, masks8, trans_params, source_image = load_bundle(stem_dir)
        if tex_src is None or not Path(tex_src).exists():
            print(f"  ! PNG texture non trovata per {stem}, salto.")
            continue

        print(f"  Solver={solver} | beta_local={beta_local} beta_global={beta_global} | auto_strength={auto_strength}")
        V_new, gamma, face_scale = morph_fused_solver(
            V, U_aligned, F, UV, masks8,
            beta_local=float(beta_local),
            beta_global=float(beta_global),
            t_low=float(t_low),
            t_high=float(t_high),
            mouth_gain=float(mouth_gain),
            eyes_gain=float(eyes_gain),
            nose_gain=float(nose_gain),
            brows_gain=float(brows_gain),
            focus_power=float(focus_power),
            global_inside=float(global_inside),
            global_outside=float(global_outside),
            solver=str(solver),
            arap_w_base=float(arap_w),
            arap_iters=int(arap_iters),
            smooth_disp_alpha=float(smooth_disp_alpha),
            auto_strength=bool(auto_strength),
            anon_ratio=float(anon_ratio),
        )
        print(f"  -> adattamento forza: gamma={gamma:.3f} | face_scale={face_scale:.3f}")

        # salva mesh morphed
        out_mesh_dir = out_dir / stem / "mesh"
        out_mesh_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_mesh_dir / "verts_morphed.npy", V_new.astype(np.float32))

        # salva OBJ+MTL con la texture originale (poi aggiorniamo al re-bake)
        out_obj = out_dir / stem / f"{stem}_morphed.obj"
        save_obj_with_uv_texture(out_obj, V_new, F, UV, tex_src, flip_v=bool(flip_v))
        print(f"  -> salvato OBJ: {out_obj}")

        # re-bake UV
        if int(rebake) == 1:
            rebaked_png = out_obj.with_name(out_obj.stem + "_rebaked.png")
            png_path = rebake_uv_png(stem_dir, V, F, UV, rebaked_png, uv_res=int(rebake_uv_res), sharpen=int(sharpen), eye_black=int(eye_black))
            if png_path:
                _rewrite_mtl_to_texture(out_obj, png_path)
                print(f"  -> MTL aggiornato: {Path(png_path).name}")
            else:
                print("  -> re-bake saltato")


def main():
    ap = argparse.ArgumentParser(description="Fase 3 — Anonimizzazione potenziata (cot/biharmonic/arap) + re-bake UV + eye-black")
    ap.add_argument("--bundles_dir", default="stage3_bundles")
    ap.add_argument("--out_dir", default="stage3_morphed")

    # morph
    ap.add_argument("--beta_local", type=float, default=0.90)
    ap.add_argument("--beta_global", type=float, default=0.35)
    ap.add_argument("--t_low", type=float, default=0.10)
    ap.add_argument("--t_high", type=float, default=0.85)
    ap.add_argument("--mouth_gain", type=float, default=1.50)
    ap.add_argument("--solver", choices=["harmonic", "biharmonic", "arap"], default="arap")
    ap.add_argument("--arap_w", type=float, default=45.0)
    ap.add_argument("--arap_iters", type=int, default=10)
    ap.add_argument("--smooth_disp_alpha", type=float, default=0.12)
    ap.add_argument("--auto_strength", type=int, default=1)
    ap.add_argument("--anon_ratio", type=float, default=0.085, help="target displacement ≈ anon_ratio * face_scale (auto_strength=1)")
    ap.add_argument("--eyes_gain", type=float, default=1.25)
    ap.add_argument("--nose_gain", type=float, default=1.15)
    ap.add_argument("--brows_gain", type=float, default=1.00)
    ap.add_argument("--focus_power", type=float, default=1.8, help=">1 = restringe la regione locale")
    ap.add_argument("--global_inside", type=float, default=0.55, help="peso globale dentro alle regioni")
    ap.add_argument("--global_outside", type=float, default=0.12, help="peso globale fuori dalle regioni")

    # re-bake
    ap.add_argument("--rebake", type=int, default=1)
    ap.add_argument("--rebake_uv_res", type=int, default=2048)
    ap.add_argument("--sharpen", type=int, default=1)
    ap.add_argument("--eye_black", type=int, default=1, help="oscura la textura nelle zone degli occhi")
    ap.add_argument("--flip_v", type=int, default=0)

    args = ap.parse_args()

    process_all(
        bundles_dir=args.bundles_dir,
        out_dir=args.out_dir,
        beta_local=args.beta_local,
        beta_global=args.beta_global,
        t_low=args.t_low,
        t_high=args.t_high,
        mouth_gain=args.mouth_gain,
        eyes_gain=args.eyes_gain,
        nose_gain=args.nose_gain,
        brows_gain=args.brows_gain,
        focus_power=args.focus_power,
        global_inside=args.global_inside,
        global_outside=args.global_outside,
        solver=args.solver,
        arap_w=args.arap_w,
        arap_iters=args.arap_iters,
        smooth_disp_alpha=args.smooth_disp_alpha,
        auto_strength=bool(args.auto_strength),
        anon_ratio=args.anon_ratio,
        rebake=args.rebake,
        rebake_uv_res=args.rebake_uv_res,
        sharpen=args.sharpen,
        eye_black=args.eye_black,
        flip_v=bool(args.flip_v)
    )


if __name__ == "__main__":
    main()
