from __future__ import annotations
import os, glob, argparse
import numpy as np
import cv2

# ---- helper ---------------------------------------------------------------

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def strict_image_from_images_root(images_root: str, stem: str) -> str | None:
    """Cerca ESCLUSIVAMENTE in images_root: <stem>.jpg|jpeg|png|bmp (non ricorsivo)."""
    for ext in EXTS:
        p = os.path.join(images_root, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def load_seg_visible_labels_full(subdir: str, stem: str):
    """Carica <stem>_projections.npz e ritorna (labels(H,W) uint8, img_size(W,H) o None)."""
    npz_path = os.path.join(subdir, f"{stem}_projections.npz")
    if not os.path.isfile(npz_path):
        return None, None, f"[skip] projections NPZ mancante: {npz_path}"

    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, None, f"[skip] NPZ non leggibile: {npz_path} ({e})"

    if 'seg_visible_labels_full' not in z.files:
        return None, None, f"[skip] chiave 'seg_visible_labels_full' assente in {npz_path}"

    labels = z['seg_visible_labels_full']
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        return None, None, f"[skip] 'seg_visible_labels_full' non è una mappa 2D in {npz_path}"
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8)

    u = np.unique(labels)
    if u.min() < 0 or u.max() > 8:
        return None, None, f"[skip] 'seg_visible_labels_full' ha valori fuori 0..8 in {npz_path}"

    img_size = None
    if 'img_size' in z.files:
        s = np.asarray(z['img_size']).reshape(-1)
        if s.size >= 2:
            img_size = (int(s[0]), int(s[1]))  # (W,H)

    return labels, img_size, None

# ---- core -----------------------------------------------------------------

def process_subdir(subdir: str, out_root: str, images_root: str,
                   remove_mouth: bool, margin_px: int, feather_px: float) -> str:
    # stem da *_extractTex.obj, altrimenti nome cartella
    objs = glob.glob(os.path.join(subdir, '*_extractTex.obj'))
    stem = os.path.splitext(os.path.basename(objs[0]))[0].replace('_extractTex', '') if objs else os.path.basename(subdir)

    # 1) labels visibili (OBBLIGATORIE)
    labels, img_size_meta, err = load_seg_visible_labels_full(subdir, stem)
    if err:
        return err
    Hlab, Wlab = labels.shape

    # 2) immagine SOLO da images_root (stesso stem)
    img_path = strict_image_from_images_root(images_root, stem)
    if img_path is None:
        return f"[skip] immagine '{stem}{EXTS}' non trovata in {images_root}"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"[skip] immagine non leggibile: {img_path}"

    H, W = img.shape[:2]
    if (W, H) != (Wlab, Hlab):
        return f"[skip] dimensioni immagine ({W}x{H}) != seg_visible ({Wlab}x{Hlab}) per '{stem}'"

    # 3) maschera di rimozione
    eyes = (labels == 1) | (labels == 2)                         # occhi
    mouth = (labels == 6) | (labels == 7) if remove_mouth else np.zeros_like(labels, dtype=bool)
    remove = eyes | mouth

    remove_u8 = (remove.astype(np.uint8) * 255)

    # dilatazione (margine aggiuntivo)
    if margin_px > 0:
        k = max(1, int(margin_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        remove_u8 = cv2.dilate(remove_u8, kernel)

    # alpha: 0 dove rimuovere, 255 altrove
    alpha = np.ones((H, W), np.uint8) * 255
    alpha[remove_u8 > 0] = 0

    # feather (opzionale)
    if feather_px > 0:
        inv = (alpha == 0).astype(np.uint8) * 255
        inv = cv2.GaussianBlur(inv.astype(np.float32), (0, 0), float(feather_px))
        alpha = (255 - inv).clip(0, 255).astype(np.uint8)

    # 4) INPAINT dei canali RGB nelle zone rimosse + compositing feather, poi salva PNG (RGB + alpha)
    #    - inpaint elimina ogni traccia nei canali colore
    #    - feather usa 'alpha' come maschera morbida per transizioni pulite
    inpaint_radius = max(3, int(round(margin_px)))  # lega il raggio al margine (min 3)
    img_inpaint = cv2.inpaint(img, remove_u8, inpaint_radius, cv2.INPAINT_TELEA)

    # Compositing: alpha=255 => tieni immagine originale; alpha=0 => usa inpaint
    soft_keep = (alpha.astype(np.float32) / 255.0)[..., None]  # (H,W,1)
    img_out = (soft_keep * img.astype(np.float32) + (1.0 - soft_keep) * img_inpaint.astype(np.float32)).astype(np.uint8)

    # Salva un solo PNG RGBA: canali RGB già ripuliti, alpha allineato a 'alpha' (con feather)
    os.makedirs(out_root, exist_ok=True)
    rgba = np.dstack([img_out, alpha])
    out_png = os.path.join(out_root, f"{stem}_masked.png")
    cv2.imwrite(out_png, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    removed_ratio = float((alpha == 0).mean())
    note = f" | removed={removed_ratio:.3f} | mouth={'on' if remove_mouth else 'off'}"
    if img_size_meta is not None:
        note += f" | img_size npz={img_size_meta[0]}x{img_size_meta[1]}"
    return f"[ok] {out_png}{note}"

# ---- main -----------------------------------------------------------------

def main():
    global in_root
    ap = argparse.ArgumentParser(description="Rimuovi (alpha=0) occhi e (opz.) bocca usando SOLO 'seg_visible_labels_full'.")
    ap.add_argument('--in_root', required=True, help='Cartella risultati 3DDFA-V3 (sottocartelle)')
    ap.add_argument('--out_root', required=True, help='Cartella di output (flat, un PNG per sottocartella)')
    ap.add_argument('--images_root', required=True, help='Cartella con le immagini originali (stesso stem, non ricorsivo)')
    ap.add_argument('--remove_mouth', type=int, default=1, help='1=rimuovi bocca (classi 6,7), 0=non rimuovere')
    ap.add_argument('--margin_px', type=int, default=0, help='Dilatazione aree rimosse (px)')
    ap.add_argument('--feather_px', type=float, default=1.0, help='Sigma feather sui bordi (0=off)')
    args = ap.parse_args()

    in_root = args.in_root
    subs = [p for p in glob.glob(os.path.join(in_root, '*')) if os.path.isdir(p)]
    if not subs:
        print(f"Nessuna sottocartella in {in_root}")
        return

    for sd in sorted(subs):
        print(process_subdir(sd, args.out_root, args.images_root,
                             remove_mouth=bool(args.remove_mouth),
                             margin_px=args.margin_px,
                             feather_px=args.feather_px))

if __name__ == '__main__':
    main()
