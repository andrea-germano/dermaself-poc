import json
import re
from pathlib import Path
import argparse
import numpy as np
import cv2

# ====== OBJ parser (v, vt, f v/vt) ======
OBJ_V_RE  = re.compile(r"^v\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*$")
OBJ_VT_RE = re.compile(r"^vt\s+([-\d.eE]+)\s+([-\d.eE]+)")
OBJ_F_RE  = re.compile(r"^f\s+(\d+)\/(\d+)\s+(\d+)\/(\d+)\s+(\d+)\/(\d+)")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_obj_with_uv(obj_path: Path):
    verts, vts, faces_v, faces_vt = [], [], [], []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                m = OBJ_V_RE.match(line)
                if m: verts.append([float(m[1]), float(m[2]), float(m[3])])
            elif line.startswith("vt "):
                m = OBJ_VT_RE.match(line)
                if m: vts.append([float(m[1]), float(m[2])])
            elif line.startswith("f "):
                m = OBJ_F_RE.match(line)
                if m:
                    faces_v.append([int(m[1])-1, int(m[3])-1, int(m[5])-1])
                    faces_vt.append([int(m[2])-1, int(m[4])-1, int(m[6])-1])
    if not verts or not vts or not faces_v:
        raise ValueError(f"OBJ incompleto: {obj_path}")
    V  = np.asarray(verts,   dtype=np.float32)
    VT = np.asarray(vts,     dtype=np.float32)
    Fv = np.asarray(faces_v, dtype=np.int32)
    Fvt= np.asarray(faces_vt,dtype=np.int32)

    # preferiamo uv per-vertice 1:1; se non c'è, media dei vt adiacenti
    if len(V) == len(VT):
        uv_coords = VT
    else:
        uv_coords = np.zeros((len(V),2), np.float32)
        counts = np.zeros((len(V),), np.int32)
        for (a,b,c),(ta,tb,tc) in zip(Fv,Fvt):
            uv_coords[a]+=VT[ta]; counts[a]+=1
            uv_coords[b]+=VT[tb]; counts[b]+=1
            uv_coords[c]+=VT[tc]; counts[c]+=1
        counts[counts==0]=1
        uv_coords/=counts[:,None]
    return V, Fv, uv_coords

# ====== rasterizzazione maschere UV da annotation_tri ======
def rasterize_region_uv(uv_coords01, tri, tri_indices_or_faces, uv_size=2048, flip_v=False):
    """
    uv_coords01: (N,2) in [0,1]
    tri: (F,3) int (0-based) - triangolazione completa
    tri_indices_or_faces:
        * array/lista di INDICI di triangoli (0-based) nella mesh globale
        * oppure array Mx3 di FACCE già pronte (come in 'annotation_tri')
    """
    H = W = int(uv_size)
    uv = uv_coords01[:, :2].astype(np.float32).copy()
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]
    px = np.column_stack([uv[:, 0] * (W - 1), uv[:, 1] * (H - 1)]).astype(np.float32)

    mask = np.zeros((H, W), dtype=np.uint8)

    tri_arr = np.asarray(tri_indices_or_faces)
    is_faces = (tri_arr.ndim == 2 and tri_arr.shape[1] == 3)

    faces_iter = tri_arr if is_faces else tri[tri_arr.astype(np.int64)]

    for face in faces_iter:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        if i < 0 or j < 0 or k < 0 or i >= len(px) or j >= len(px) or k >= len(px):
            continue
        poly = np.array([px[i], px[j], px[k]], dtype=np.int32)
        if not np.isfinite(poly).all():
            continue
        cv2.fillConvexPoly(mask, poly, 255)  # triangolo convesso → ok
    return mask

# ====== pesi per-vertice da mask UV ======
def weights_from_mask_uv(uv_coords01, mask_uv):
    H, W = mask_uv.shape[:2]
    u = np.clip(uv_coords01[:,0]*(W-1), 0, W-1)
    v = np.clip(uv_coords01[:,1]*(H-1), 0, H-1)
    x0 = np.floor(u).astype(int); x1 = np.clip(x0+1, 0, W-1)
    y0 = np.floor(v).astype(int); y1 = np.clip(y0+1, 0, H-1)
    Ia = mask_uv[y0, x0].astype(np.float32)
    Ib = mask_uv[y1, x0].astype(np.float32)
    Ic = mask_uv[y0, x1].astype(np.float32)
    Id = mask_uv[y1, x1].astype(np.float32)
    wa = (x1 - u)*(y1 - v)
    wb = (x1 - u)*(v - y0)
    wc = (u - x0)*(y1 - v)
    wd = (u - x0)*(v - y0)
    s = (wa*Ia + wb*Ib + wc*Ic + wd*Id)/255.0
    return np.clip(s,0.0,1.0).astype(np.float32)

# ====== Procrustes pesato (src->dst) ======
def procrustes_align(src_xyz, dst_xyz, weights=None):
    if weights is None:
        ws = np.ones((src_xyz.shape[0],1), np.float64)
    else:
        ws = weights.reshape(-1,1).astype(np.float64)
    wsum = ws.sum()
    mu_s = (ws*src_xyz).sum(0)/wsum
    mu_d = (ws*dst_xyz).sum(0)/wsum
    X = src_xyz - mu_s
    Y = dst_xyz - mu_d
    Xw = X*ws
    U,S,Vt = np.linalg.svd(Xw.T@Y, full_matrices=False)
    R = Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1
        R = Vt.T@U.T
    num = np.trace((Xw.T@Y)@R.T)
    den = (ws*(X**2)).sum()
    s = float(num/den) if den>0 else 1.0
    t = mu_d - s*(R@mu_s)
    src_aligned = (s*(src_xyz@R.T))+t
    return src_aligned, (s,R,t)

# ====== helper: trova file per stem ======
def find_by_stem(root: Path, stem: str, exts):
    """Cerca ricorsivamente un file con nome base == stem e una delle estensioni in 'exts'."""
    stem_lc = stem.lower()
    for p in root.rglob("*"):
        if not p.is_file(): 
            continue
        if p.suffix.lower() in exts and p.stem.lower() == stem_lc:
            return p
    return None

def main(input_root="examples/results",
         assets_dir="assets",
         out_root="stage3_bundles",
         images_root=None,
         trans_params_root=None,
         uv_size=0,
         flip_v=False):

    input_root = Path(input_root)
    assets_dir = Path(assets_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # face model (triangolazione + annotazioni + mean)
    model = np.load(str(assets_dir/"face_model.npy"), allow_pickle=True).item()
    tri = model['tri'].astype(np.int32)
    ann_tri = model['annotation_tri']  # lista: ogni entry può essere [indici] oppure array Facce Mx3
    u_mean = model['u'].astype(np.float32).reshape(-1,3)

    # trova TUTTI i *_extractTex.obj in modo ricorsivo
    obj_list = list(input_root.rglob("*_extractTex.obj"))
    if not obj_list:
        print(f"Nessun *_extractTex.obj trovato in {input_root}")
        return

    for obj in sorted(obj_list):
        stem = obj.stem.replace("_extractTex","")
        tex = obj.with_suffix(".png")
        if not tex.exists():
            print(f"[{stem}] PNG texture mancante accanto all'OBJ, salto")
            continue

        # parse mesh + UV dall'OBJ
        try:
            V, F, UV = parse_obj_with_uv(obj)
        except Exception as e:
            print(f"[{stem}] OBJ non leggibile ({e}), salto")
            continue

        # UV size
        if uv_size and uv_size > 0:
            uv_res = int(uv_size)
        else:
            tex_img = cv2.imread(str(tex), cv2.IMREAD_COLOR)
            if tex_img is None:
                print(f"[{stem}] PNG non leggibile, uso uv_res=2048 di default")
                uv_res = 2048
            else:
                H,W = tex_img.shape[:2]
                uv_res = max(H,W)

        # ===== Maschere UV coarse-8 =====
        mask_dir = out_root/f"{stem}"/"masks_uv"
        mask_dir.mkdir(parents=True, exist_ok=True)
        masks8 = []
        for ridx, tri_idx in enumerate(ann_tri):
            m = rasterize_region_uv(UV, tri, tri_idx, uv_size=uv_res, flip_v=flip_v)
            cv2.imwrite(str(mask_dir/f"mask_region{ridx+1}.png"), m, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            masks8.append(m)
        masks8 = np.stack(masks8, axis=-1)  # HxWx8
        np.save(str(mask_dir/"masks_uv_coarse8.npy"), masks8)

        # ===== Pesi per-vertice utili (debug/QA) =====
        mask_eyes = np.maximum(masks8[...,0], masks8[...,1])
        mask_lips = np.maximum(masks8[...,5], masks8[...,6])
        mask_skin = masks8[...,7]
        W_eyes = weights_from_mask_uv(UV, mask_eyes)
        W_lips = weights_from_mask_uv(UV, mask_lips)
        W_skin = weights_from_mask_uv(UV, mask_skin)
        W_target = np.maximum(W_eyes, W_lips).astype(np.float32)
        np.save(str(mask_dir/"weights_eyes.npy"),   W_eyes)
        np.save(str(mask_dir/"weights_lips.npy"),   W_lips)
        np.save(str(mask_dir/"weights_skin.npy"),   W_skin)
        np.save(str(mask_dir/"weights_target.npy"), W_target)

        # ===== Procrustes (mean -> V) pesato su pelle =====
        U_aligned, (s,R,t) = procrustes_align(u_mean.astype(np.float64), V.astype(np.float64), weights=W_skin.astype(np.float64))

        # ===== Scrittura mesh/ ====
        mesh_dir = out_root/f"{stem}"/"mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(mesh_dir/"verts.npy"),     V.astype(np.float32))
        np.save(str(mesh_dir/"faces.npy"),     F.astype(np.int32))
        np.save(str(mesh_dir/"uv_coords.npy"), UV.astype(np.float32))
        np.save(str(mesh_dir/"u_mean.npy"),    u_mean.astype(np.float32))
        with open(mesh_dir/"procrustes.json","w",encoding="utf-8") as f:
            json.dump({"s": s, "R": R.tolist(), "t": t.tolist()}, f, ensure_ascii=False, indent=2)

        # ===== Manifest + source_image (se riusciamo a trovarla) =====
        manifest = {
            "uv_size": uv_res,
            "uv_convention": "u-right_v-down" if not flip_v else "u-right_v-up",
            "faces_are_0_based": True,
            "source_obj": str(obj),
            "source_tex": str(tex)
        }

        if images_root:
            img_root = Path(images_root)
            img_path = find_by_stem(img_root, stem, IMG_EXTS)
            if img_path:
                manifest["source_image"] = str(img_path.resolve())

        with open(out_root/f"{stem}"/"manifest.json","w",encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # ===== trans_params.npy (se disponibile da una radice separata) =====
        if trans_params_root:
            troot = Path(trans_params_root)
            # convenzioni cercate: <stem>_trans_params.npy  oppure  trans_params_<stem>.npy
            cand1 = find_by_stem(troot, stem + "_trans_params", {".npy"})
            cand2 = find_by_stem(troot, "trans_params_" + stem, {".npy"})
            trans_src = cand1 if cand1 else cand2
            if trans_src and trans_src.exists():
                dst = mesh_dir/"trans_params.npy"
                try:
                    arr = np.load(str(trans_src))
                    if arr.size == 5:
                        np.save(str(dst), arr.astype(np.float32))
                        print(f"[{stem}] copiato trans_params.npy da {trans_src}")
                    else:
                        print(f"[{stem}] {trans_src} ha shape inattesa (attesi 5 valori)")
                except Exception as e:
                    print(f"[{stem}] impossibile leggere {trans_src}: {e}")

        print(f"[OK] bundle → {out_root/stem}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default="examples/results",
                    help="Cartella radice dei risultati 3DDFA-V3 (scansione ricorsiva)")
    ap.add_argument("--assets_dir", default="assets")
    ap.add_argument("--out_root", default="stage3_bundles")
    ap.add_argument("--images_root", default="", help="(opzionale) radice in cui cercare le foto originali per source_image")
    ap.add_argument("--trans_params_root", default="", help="(opzionale) radice in cui cercare i file *_trans_params.npy")
    ap.add_argument("--uv_size", type=int, default=0, help="0=usa risoluzione della PNG, altrimenti forza (es. 2048)")
    ap.add_argument("--flip_v", type=int, default=0, help="1 per flip V durante rasterizzazione UV")
    args = ap.parse_args()

    main(
        input_root=args.input_root,
        assets_dir=args.assets_dir,
        out_root=args.out_root,
        images_root=(args.images_root if args.images_root else None),
        trans_params_root=(args.trans_params_root if args.trans_params_root else None),
        uv_size=args.uv_size,
        flip_v=bool(args.flip_v)
    )
