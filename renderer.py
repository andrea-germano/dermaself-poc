import numpy as np, torch, imageio.v2 as imageio
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    RasterizationSettings, OrthographicCameras, AmbientLights, BlendParams
)

OBJ_PATH = "output2/edo/edo.obj"
OUT_PATH = "front.png"
IMG_SIZE = 1024
MARGIN = 0.95     # quanto riempire il frame
BG_WHITE = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_numpy_img(rgba, bg_white=True):
    rgb, a = rgba[..., :3], rgba[..., 3:4]
    if bg_white:
        out = rgb * a + (1 - a)
        return (out.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    else:
        return (rgba.clamp(0,1).cpu().numpy()*255).astype(np.uint8)

# Carica mesh (UV + texture da .mtl)
mesh = load_objs_as_meshes([OBJ_PATH], device=device)

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
    pts = cam.transform_points_screen(V, image_size=((IMG_SIZE, IMG_SIZE),))
    xy = pts[..., :2]
    xy_min, xy_max = xy.amin(1), xy.amax(1)
    w0 = (xy_max[0,0] - xy_min[0,0]).clamp(min=1.0)
    h0 = (xy_max[0,1] - xy_min[0,1]).clamp(min=1.0)
    s_fit = float(MARGIN * IMG_SIZE / max(w0.item(), h0.item()))

cam = OrthographicCameras(
    device=device, R=R_cam, T=T_cam,
    focal_length=((s_fit, s_fit),), principal_point=((0.0, 0.0),), in_ndc=True
)

blend = BlendParams(sigma=1e-8, gamma=1e-8, background_color=(1.0, 1.0, 1.0))

shader = SoftPhongShader(device=device, cameras=cam, lights=AmbientLights(device=device, ambient_color=((1,1,1),)),
                         blend_params=blend)

# Unlit: solo componente ambient -> “texture-only”
lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))
rast = RasterizationSettings(image_size=IMG_SIZE, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cam, raster_settings=rast),
    shader=shader
)

# --- render ---
rgba = renderer(mesh)[0]           # (H, W, 4) in [0,1]

# Alpha binario (0/1). Puoi regolare la soglia (0.3–0.7) in base ai bordi.
a = (rgba[..., 3:4] > 0.5).float()

# 1) PNG con trasparenza netta (senza semi-trasparenze)
png_rgba = torch.cat([rgba[..., :3], a], dim=-1)
img_rgba = (png_rgba.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
imageio.imwrite(OUT_PATH, img_rgba)