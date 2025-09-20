from __future__ import annotations
import argparse, os, sys, glob
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Mesh I/O con UV (senza fallback)
# -----------------------------
@dataclass
class Mesh:
    V: np.ndarray         # (N,3) float32
    F: np.ndarray         # (M,3) int32 0-based
    UV: np.ndarray        # (K,2) float32 in [0,1]
    F_uv: np.ndarray      # (M,3) int32 0-based, per-vertex uv indices

def load_obj_with_uv(path: str) -> Mesh:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OBJ non trovato: {path}")
    vs, vts, faces_v, faces_vt = [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()[:4]
                vs.append([float(x), float(y), float(z)])
            elif line.startswith('vt '):
                parts = line.strip().split()
                if len(parts) < 3:
                    raise ValueError(f"Riga vt malformata in {path}")
                _, u, v = parts[:3]
                vts.append([float(u), float(v)])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                idxs = [p.split('/') for p in parts]
                if len(idxs) != 3:
                    raise ValueError("L'OBJ deve essere triangolato (facce a 3 vertici).")
                v_idx, vt_idx = [], []
                for t in idxs:
                    if len(t) < 2 or t[1] == '':
                        raise ValueError("Le facce devono avere indici UV (formato v/vt).")
                    v_i = int(t[0]); vt_i = int(t[1])
                    v_idx.append(v_i-1 if v_i > 0 else v_i)
                    vt_idx.append(vt_i-1 if vt_i > 0 else vt_i)
                faces_v.append(v_idx); faces_vt.append(vt_idx)
    if not vs:  raise ValueError("OBJ senza vertici.")
    if not vts: raise ValueError("OBJ senza coordinate UV (vt).")
    if not faces_v: raise ValueError("OBJ senza facce.")
    V   = np.asarray(vs, dtype=np.float32)
    F   = np.asarray(faces_v, dtype=np.int32)
    Fuv = np.asarray(faces_vt, dtype=np.int32)
    UV  = np.asarray(vts, dtype=np.float32)
    return Mesh(V=V, F=F, UV=UV, F_uv=Fuv)

def save_obj_with_uv(path_obj: str, mesh: Mesh, texture_path: str):
    os.makedirs(os.path.dirname(path_obj), exist_ok=True)
    base = os.path.splitext(os.path.basename(path_obj))[0]
    mtl_name = base + '.mtl'
    with open(path_obj, 'w', encoding='utf-8') as f:
        f.write(f"mtllib {mtl_name}\n")
        for v in mesh.V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vt in mesh.UV:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        f.write("usemtl material_0\n")
        for (vi, vti) in zip(mesh.F, mesh.F_uv):
            a, b, c = vi + 1
            ta, tb, tc = vti + 1
            f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
    mtl_path = os.path.join(os.path.dirname(path_obj), mtl_name)
    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\nTr 1.000\nillum 1\nNs 0.000\n")
        texfile = os.path.basename(texture_path)
        f.write(f"map_Kd {texfile}\n")

# -----------------------------
# Morph verso mean face (Umeyama)
# -----------------------------
def umeyama_alignment(X: np.ndarray, Y: np.ndarray):
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch in Umeyama: {X.shape} vs {Y.shape}")
    n = X.shape[0]
    muX = X.mean(axis=0); muY = Y.mean(axis=0)
    Xc = X - muX; Yc = Y - muY
    C = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varX = (Xc**2).sum() / n
    s = np.trace(np.diag(D) @ S) / varX
    t = muY - s * (R @ muX)
    return R, s, t

def blend_to_mean(V: np.ndarray, V_mean: np.ndarray, alpha: float, keep_pose: bool=True) -> np.ndarray:
    if V.shape != V_mean.shape:
        raise ValueError(f"Shape mismatch: V {V.shape} vs V_mean {V_mean.shape}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('alpha deve essere in [0,1]')
    if keep_pose:
        R, s, t = umeyama_alignment(V, V_mean)
        V_aligned = (s * (R @ V.T)).T + t
        V_blend   = (1 - alpha) * V_aligned + alpha * V_mean
        V_out     = (R.T @ ((V_blend - t).T) / s).T
    else:
        V_out = (1 - alpha) * V + alpha * V_mean
    return V_out.astype(np.float32)

# -----------------------------
# Mean face loader (solo chiave 'u')
# -----------------------------
def load_mean_vertices_from_face_model(target_N: int, face_model_path: str) -> np.ndarray:
    if not os.path.isfile(face_model_path):
        raise FileNotFoundError(f"face_model non trovato: {face_model_path}")
    arr = np.load(face_model_path, allow_pickle=True)
    # Supporta: npz con 'u' oppure npy contenente dict con 'u'
    if isinstance(arr, np.lib.npyio.NpzFile):
        if 'u' not in arr.files:
            raise KeyError("Nel face_model manca la chiave 'u'.")
        u = arr['u']
    elif isinstance(arr, np.ndarray) and getattr(arr, 'dtype', None) == object:
        d = arr.item()
        if not isinstance(d, dict) or 'u' not in d:
            raise KeyError("Nel face_model .npy atteso un dict con chiave 'u'.")
        u = d['u']
    else:
        raise KeyError("Nel face_model .npy atteso un dict con chiave 'u'.")
    u = np.asarray(u).astype(np.float32).reshape(-1)
    if u.size % 3 != 0:
        raise ValueError(f"Dimensione 'u' non multipla di 3: {u.size}")
    N = u.size // 3
    if N != target_N:
        raise ValueError(f"Mean face ha N={N} vertici, ma la mesh ha N={target_N}.")
    return u.reshape(-1, 3)

# -----------------------------
# v2d loader (senza fallback): prende *solo* da <stem>_projections.npz, chiave 'v2d_full'
# -----------------------------
def load_v2d(subdir: str, stem: str, expected_N: int) -> np.ndarray:
    npz_path = os.path.join(subdir, f"{stem}_projections.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"File non trovato: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    if 'v2d_full' not in z.files:
        raise KeyError(f"{npz_path} non contiene la chiave 'v2d_full'.")
    v = z['v2d_full']
    if not (isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2):
        raise ValueError(f"'v2d_full' deve essere (N,2). Trovato {getattr(v,'shape',None)}")
    if v.shape[0] != expected_N:
        raise ValueError(f"'v2d_full' ha N={v.shape[0]} ma la mesh ha N={expected_N}.")
    return v.astype(np.float32)

# -----------------------------
# Raster & Baking
# -----------------------------
def warp_triangle(src_img, dst_img, tri_src, tri_dst, z, zbuf):
    tri_src = tri_src.astype(np.float32)
    tri_dst = tri_dst.astype(np.float32)
    r1 = cv2.boundingRect(tri_src)
    r2 = cv2.boundingRect(tri_dst)
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    tri_src_c = np.array([[tri_src[0][0]-r1[0], tri_src[0][1]-r1[1]],
                          [tri_src[1][0]-r1[0], tri_src[1][1]-r1[1]],
                          [tri_src[2][0]-r1[0], tri_src[2][1]-r1[1]]], dtype=np.float32)
    tri_dst_c = np.array([[tri_dst[0][0]-r2[0], tri_dst[0][1]-r2[1]],
                          [tri_dst[1][0]-r2[0], tri_dst[1][1]-r2[1]],
                          [tri_dst[2][0][0]-r2[0] if isinstance(tri_dst[2][0], (list, np.ndarray)) else tri_dst[2][0]-r2[0], 
                           tri_dst[2][1][0]-r2[1] if isinstance(tri_dst[2][1], (list, np.ndarray)) else tri_dst[2][1]-r2[1]]], dtype=np.float32)
    src_roi = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_roi.size == 0: return
    M = cv2.getAffineTransform(tri_src_c, tri_dst_c)
    warped = cv2.warpAffine(src_roi, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_c), 1.0)
    dst_roi = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    z_roi  = zbuf[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    update = (mask > 0) & ((z < z_roi) | (z_roi == 0))
    if not np.any(update): return
    ys, xs = np.nonzero(update)
    dst_roi[ys, xs, :] = warped[ys, xs, :]
    z_roi[ys, xs] = z
    dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_roi
    zbuf[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]   = z_roi

def bake_texture_from_v2d(image_bgr: np.ndarray, mesh: Mesh, v2d: np.ndarray, tex_size: int) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    atlas = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    zbuf  = np.zeros((tex_size, tex_size), dtype=np.float32)
    # Flip asse V delle UV (OBJ: origine in basso; immagini: origine in alto)
    uv = mesh.UV.astype(np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    uv_px_all = uv * np.array([tex_size-1, tex_size-1], dtype=np.float32)
    # Ordina facce far->near (usa Z dei vertici come proxy)
    face_depth = mesh.V[mesh.F, 2].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    for fi in order:
        f = mesh.F[fi]; fuv = mesh.F_uv[fi]
        tri_img = v2d[f]  # (3,2)
        x, y = tri_img[:, 0], tri_img[:, 1]
        if (x.max() < 0 or x.min() >= W or y.max() < 0 or y.min() >= H): 
            continue
        tri_src = np.stack([np.clip(x, 0, W-1), np.clip(y, 0, H-1)], axis=1)
        tri_uv  = uv_px_all[fuv]
        z = face_depth[fi]
        warp_triangle(image, atlas, tri_src, tri_uv, z, zbuf)
    return atlas

# -----------------------------
# Utility immagini (senza euristiche): nome file = <stem> con estensione in {png,jpg,jpeg}
# -----------------------------
def expected_image_path(images_root: str, stem: str) -> str:
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"--images_root non esiste: {images_root}")
    exts = {'.png', '.jpg', '.jpeg'}
    stem_low = stem.lower()

    matches = []
    for name in os.listdir(images_root):
        path = os.path.join(images_root, name)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() in exts and base.lower().startswith(stem_low):
            matches.append(path)

    if not matches:
        expected = ", ".join([os.path.join(images_root, stem + e) for e in ['.png', '.jpg', '.jpeg']])
        raise FileNotFoundError(
            f"Immagine non trovata per '{stem}'. Cercati (in --images_root): {expected} "
            f"oppure un file che INIZI con '{stem}' e sia PNG/JPG/JPEG."
        )

    # Se c’è una corrispondenza esatta del basename (case-insensitive), usala.
    exact = [p for p in matches if os.path.splitext(os.path.basename(p))[0].lower() == stem_low]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"Trovate più immagini con basename esatto '{stem}'. "
            "Tieni un solo file: " + ", ".join(sorted(os.path.basename(x) for x in exact))
        )

    # Altrimenti, più file che iniziano con stem: segnala ambiguità (no scelte euristiche).
    if len(matches) > 1:
        raise ValueError(
            f"Trovate più immagini che iniziano con '{stem}'. "
            "Tieni un solo file: " + ", ".join(sorted(os.path.basename(x) for x in matches))
        )

    return matches[0]

# -----------------------------
# Elaborazione singola persona
# -----------------------------
def process_one(obj_path: str, face_model_path: str, image_path: str, out_dir: str, alpha: float, tex_size: int, keep_pose: bool):
    mesh = load_obj_with_uv(obj_path)
    V_mean = load_mean_vertices_from_face_model(mesh.V.shape[0], face_model_path)

    subdir = os.path.dirname(obj_path)
    stem   = os.path.splitext(os.path.basename(obj_path))[0].replace('_extractTex','')

    # 1) carica v2d (in pixel originali) — obbligatorio e solo da v2d_full
    v2d = load_v2d(subdir, stem, expected_N=mesh.V.shape[0])

    # 2) immagine originale
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Immagine non leggibile: {image_path}")

    # 3) bake con v2d (allineamento perfetto, no camera)
    tex = bake_texture_from_v2d(img_bgr, mesh, v2d, tex_size=tex_size)

    # 4) morph verso faccia media
    V_anon = blend_to_mean(mesh.V, V_mean, alpha=alpha, keep_pose=keep_pose)

    # 5) salva risultati nella sottocartella di output
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, f'{stem}_texture.png')
    Image.fromarray(tex).save(out_tex)

    out_obj = os.path.join(out_dir, f'{stem}.obj')
    mesh_out = Mesh(V=V_anon, F=mesh.F, UV=mesh.UV, F_uv=mesh.F_uv)
    save_obj_with_uv(out_obj, mesh_out, out_tex)

    return out_obj, out_tex

# -----------------------------
# Main (batch senza fallback)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description='Batch anonymize + HR rebake (v2d_full in *_projections.npz).')
    ap.add_argument('--in_root', required=True, help='Cartella con le sottocartelle (una per persona).')
    ap.add_argument('--images_root', required=True, help='Cartella contenente le immagini originali (file <stem>.png/.jpg/.jpeg).')
    ap.add_argument('--face_model', required=True, help="face_model .npy/.npz con chiave 'u' (3N x 1 o 3N).")
    ap.add_argument('--out_root', required=True, help='Cartella di output (replica la struttura delle sottocartelle di input).')
    ap.add_argument('--alpha', type=float, default=1.0, help='Intensità morph verso faccia media [0..1].')
    ap.add_argument('--tex_size', type=int, default=1024, help='Risoluzione texture UV (es. 2048, 4096).')
    ap.add_argument('--obj_glob', type=str, default='*_extractTex.obj', help='Pattern OBJ da processare in ogni sottocartella.')
    ap.add_argument('--keep_pose', action='store_true', help='Mantieni la posa originale (Umeyama).')
    args = ap.parse_args()

    # Raccogli tutti gli OBJ matching nelle sottocartelle (non elaboriamo file sciolti in in_root)
    subdirs = [p for p in glob.glob(os.path.join(args.in_root, '*')) if os.path.isdir(p)]
    if not subdirs:
        print("Nessuna sottocartella trovata in --in_root.", file=sys.stderr)
        sys.exit(1)

    obj_list: List[str] = []
    for sub in subdirs:
        obj_list.extend(glob.glob(os.path.join(sub, args.obj_glob)))
    obj_list = sorted(set(obj_list))
    if not obj_list:
        print(f"Nessun OBJ trovato con pattern {args.obj_glob} nelle sottocartelle di {args.in_root}.", file=sys.stderr)
        sys.exit(1)

    for obj_path in obj_list:
        subdir = os.path.dirname(obj_path)
        rel    = os.path.relpath(subdir, args.in_root)
        stem_ex = os.path.splitext(os.path.basename(obj_path))[0].replace('_extractTex','')
        try:
            img_path = expected_image_path(args.images_root, stem_ex)
            out_dir = os.path.join(args.out_root, rel)
            out_obj, out_tex = process_one(
                obj_path=obj_path,
                face_model_path=args.face_model,
                image_path=img_path,
                out_dir=out_dir,
                alpha=args.alpha,
                tex_size=args.tex_size,
                keep_pose=args.keep_pose,
            )
            print(f"OK: {os.path.relpath(out_obj, args.out_root)}")
        except Exception as e:
            print(f"ERRORE su {obj_path}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
