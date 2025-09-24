
import argparse, sys, numpy as np, torch, imageio.v2 as imageio
from pathlib import Path

try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        MeshRenderer, MeshRasterizer, SoftPhongShader,
        RasterizationSettings, OrthographicCameras, AmbientLights, BlendParams
    )
except Exception as e:
    print("[ERRORE] pytorch3d non è installato o non è importabile:", e, file=sys.stderr)
    print("         Installa pytorch3d (anche CPU-only) oppure salta lo Step 4.", file=sys.stderr)
    sys.exit(1)


def to_numpy_img(rgba, bg_white=True):
    rgb, a = rgba[..., :3], rgba[..., 3:4]
    if bg_white:
        out = rgb * a + (1 - a)
        return (out.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    else:
        return (rgba.clamp(0,1).cpu().numpy()*255).astype(np.uint8)


def render_obj(obj_path: Path, out_path: Path, img_size=1024, margin=0.95, bg_white=False,
               alpha_thresh=0.5, device=None):
    obj_path = Path(obj_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Carica mesh (UV + texture da .mtl)
    mesh = load_objs_as_meshes([str(obj_path)], device=device)

    # Centra in XY
    verts = mesh.verts_packed()
    center_xy = verts[:, :2].mean(0)
    mesh.offset_verts_(-torch.tensor([center_xy[0], center_xy[1], 0.0], device=device))

    # Camera ortografica frontale (R=I, T=0), zoom iniziale 1.0
    R_cam = torch.eye(3, device=device)[None]
    T_cam = torch.zeros(1, 3, device=device)
    cam = OrthographicCameras(
        device=device, R=R_cam, T=T_cam,
        focal_length=((1.0, 1.0),), principal_point=((0.0, 0.0),), in_ndc=True
    )

    # Auto-fit: misura con zoom 1 e imposta lo zoom giusto
    with torch.no_grad():
        V = mesh.verts_padded()
        pts = cam.transform_points_screen(V, image_size=((img_size, img_size),))
        xy = pts[..., :2]
        xy_min, xy_max = xy.amin(1), xy.amax(1)
        w0 = (xy_max[0,0] - xy_min[0,0]).clamp(min=1.0)
        h0 = (xy_max[0,1] - xy_min[0,1]).clamp(min=1.0)
        s_fit = float(margin * img_size / max(w0.item(), h0.item()))

    cam = OrthographicCameras(
        device=device, R=R_cam, T=T_cam,
        focal_length=((s_fit, s_fit),), principal_point=((0.0, 0.0),), in_ndc=True
    )

    blend = BlendParams(sigma=1e-8, gamma=1e-8, background_color=(1.0, 1.0, 1.0))

    shader = SoftPhongShader(
        device=device, cameras=cam,
        lights=AmbientLights(device=device, ambient_color=((1,1,1),)),
        blend_params=blend
    )

    rast = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=rast),
        shader=shader
    )

    # --- render ---
    with torch.no_grad():
        rgba = renderer(mesh)[0]  # (H, W, 4) in [0,1]

    # Alpha binario
    a = (rgba[..., 3:4] > float(alpha_thresh)).float()

    # Output
    if bg_white:
        # composita su bianco e salva RGB
        rgb = rgba[..., :3] * a + (1 - a)
        img = (rgb.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(str(out_path), img)
    else:
        # salva RGBA con alpha netto
        png_rgba = torch.cat([rgba[..., :3], a], dim=-1)
        img_rgba = (png_rgba.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(str(out_path), img_rgba)


def find_obj_for_folder(folder: Path):
    """Cerca uno .obj nella sottocartella:
       1) <folder>/<folder.name>.obj
       2) il primo .obj trovato (ordinato alfabeticamente)
    """
    candidate = folder / f"{folder.name}.obj"
    if candidate.exists():
        return candidate
    objs = sorted(folder.glob("*.obj"))
    return objs[0] if objs else None


def parse_args():
    p = argparse.ArgumentParser(description="Renderer PNG frontale per mesh (.obj) con texture (PyTorch3D).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--obj", type=str, help="Percorso a un singolo file .obj da renderizzare.")
    mode.add_argument("--in_root", type=str, help="Cartella con sottocartelle (una per immagine) contenenti .obj.")
    p.add_argument("--out", type=str, help="Percorso file PNG in uscita (solo per --obj).")
    p.add_argument("--out_root", type=str, help="Cartella output per i PNG (solo per --in_root).", default="out/render")
    p.add_argument("--img_size", type=int, default=1024, help="Lato immagine di output (px).")
    p.add_argument("--margin", type=float, default=0.95, help="Quanto riempire il frame (0..1).")
    p.add_argument("--bg_white", type=int, default=0, help="1 per sfondo bianco RGB, 0 per PNG RGBA.")
    p.add_argument("--alpha_thresh", type=float, default=0.5, help="Soglia per alpha binaria (0..1).")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 | mps (se supportato)")
    p.add_argument("--stop_on_error", action="store_true", help="In batch, ferma alla prima eccezione.")
    return p.parse_args()


def get_device(dev_str: str):
    if dev_str is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = dev_str.lower()
    if ds == "cpu":
        return torch.device("cpu")
    if ds == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if ds.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(ds)
        print("[AVVISO] CUDA non disponibile: uso CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.device)

    if args.obj:
        obj_path = Path(args.obj)
        if not obj_path.exists():
            print(f"[ERRORE] .obj non trovato: {obj_path}", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.out) if args.out else obj_path.with_suffix(".png")
        print(f"[INFO] Rendering singolo: {obj_path} -> {out_path} (device={device})")
        render_obj(obj_path, out_path, img_size=args.img_size, margin=args.margin,
                   bg_white=bool(args.bg_white), alpha_thresh=args.alpha_thresh, device=device)
        print("[OK] Salvataggio:", out_path)
        return

    # Batch: scansiona sottocartelle di in_root
    in_root = Path(args.in_root)
    if not in_root.exists():
        print(f"[ERRORE] Cartella input non trovata: {in_root}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in sorted(in_root.iterdir()) if d.is_dir()]
    if not subdirs:
        print(f"[AVVISO] Nessuna sottocartella trovata in {in_root}", file=sys.stderr)

    ok, failed = 0, 0
    for d in subdirs:
        try:
            obj_path = find_obj_for_folder(d)
            if obj_path is None:
                print(f"[SKIP] Nessun .obj in {d}", file=sys.stderr)
                failed += 1
                if args.stop_on_error:
                    sys.exit(2)
                continue
            out_path = out_root / f"{d.name}_front.png"
            print(f"[INFO] {d.name}: {obj_path.name} -> {out_path.name}")
            render_obj(obj_path, out_path, img_size=args.img_size, margin=args.margin,
                       bg_white=bool(args.bg_white), alpha_thresh=args.alpha_thresh, device=device)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[ERRORE] {d.name}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            if args.stop_on_error:
                sys.exit(3)

    print(f"[FINE] Render completato. Successi: {ok}, Falliti/Skippati: {failed}. Output in: {out_root}")


if __name__ == "__main__":
    main()
