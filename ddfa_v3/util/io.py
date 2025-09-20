import argparse
import cv2
import os
import sys
import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
# from .preprocess import back_resize_crop_img
# from .nv_diffrast import MeshRenderer

def _save_trans_and_manifest(save_path, img_name, trans_params, source_tex_path=None, source_image_path=None):
    """
    Salva:
      - <img_name>_trans_params.npy con [w0, h0, s, tx, ty] (float32)
      - manifest.json (aggiunge/aggiorna campi se già esiste)
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # estrai numeri python/float (accetta numpy scalars / torch tensors / list)
    def _to_py_float(x):
        try:
            # torch tensor
            import torch
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
        except Exception:
            pass
        # numpy scalar o python
        return float(np.array(x).astype(np.float32))

    # trans_params atteso: [w0,h0,s,tx,ty] (in 3DDFA-V3 è già così)
    w0 = _to_py_float(trans_params[0])
    h0 = _to_py_float(trans_params[1])
    s  = _to_py_float(trans_params[2])
    tx = _to_py_float(trans_params[3])
    ty = _to_py_float(trans_params[4])

    tp = np.array([w0, h0, s, tx, ty], dtype=np.float32)
    np.save(str(save_dir / f"{img_name}_trans_params.npy"), tp)

    # aggiorna/crea manifest.json
    man_path = save_dir / f"{img_name}_manifest.json"
    if man_path.exists():
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("uv_convention", "u-right_v-down")
    manifest["trans_params"] = [w0, h0, s, tx, ty]

    if source_tex_path:
        manifest["source_tex"] = str(Path(source_tex_path).resolve())
    if source_image_path:
        manifest["source_image"] = str(Path(source_image_path).resolve())

    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def plot_kpts(image, kpts, color = 'g'):

    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :]
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)

    return image

def show_seg_visble(new_seg_visible_one, img):

    img = img.copy()
    new_seg_visible_one = new_seg_visible_one.copy()
    mask2=np.stack((new_seg_visible_one[:,:,0],)*3, axis=-1).astype(np.uint8)

    back2=np.full_like(mask2,0)

    colormap = label_colormap(9)
    alphas = np.linspace(0.75, 0.25, num=9)

    dst2=np.full_like(back2,0)
    for i, mask in enumerate(mask2[:,:,0][None,:,:]):
        alpha = alphas[i]
        index = mask > 0
        res = colormap[mask]
        dst2[index] = (1 - alpha) * back2[index].astype(float) + alpha * res[index].astype(float)
    dst2 = np.clip(dst2.round(), 0, 255).astype(np.uint8)

    return ((dst2[:,:,::-1]*0.5+img*0.5)).astype(np.uint8)

def label_colormap(n_label=9):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 9).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """
    if n_label == 9:  # helen, ibugmask
        cmap = np.array(
            [
                (0, 0, 0),
                (0, 205, 0),
                (0, 138, 0),
                (139, 76, 57),
                (139, 54, 38),
                (154, 50, 205),
                (72, 118, 255),
                (22, 22, 139),
                (255, 255, 0),
            ],
            dtype=np.uint8,
        )
    else:

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap

def back_resize_crop_img(img, trans_params, ori_img, resample_method = Image.BICUBIC):
    
    w0, h0, s, t, target_size = trans_params[0], trans_params[1], trans_params[2], [trans_params[3],trans_params[4]], 224
    
    img=Image.fromarray(img)
    ori_img=Image.fromarray(ori_img)
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    old_img = ori_img
    old_img = old_img.resize((w, h), resample=resample_method)

    old_img.paste(img, (left, up, right, below))
    old_img = old_img.resize((int(w0), int(h0)), resample=resample_method)

    old_img = np.array(old_img)
    return old_img

def back_resize_ldms(ldms, trans_params):
    
    w0, h0, s, t, target_size = trans_params[0], trans_params[1], trans_params[2], [trans_params[3],trans_params[4]], 224

    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    ldms[:, 0] = ldms[:, 0] + left
    ldms[:, 1] = ldms[:, 1] + up

    ldms[:, 0] = ldms[:, 0] / w * w0
    ldms[:, 1] = ldms[:, 1] / h * h0

    return ldms

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # obj start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        s = '# Results of 3DDFA-V3, https://github.com/wang-zidu/3DDFA-V3\n'
        f.write(s)

        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)

def save_obj_with_uv_texture(obj_out_path, verts_xyz, tri, uv_coords01, tex_img_bgr_uint8):
    ''' Scrive:
    - OBJ con v + vt + f v/vt
    - MTL con map_Kd
    - PNG (texture UV)
    uv_coords01: (Nvert, 2) in [0,1]
    tri: (Ntri, 3) 0-based
    '''
    obj_out_path = Path(obj_out_path)
    mtl_out_path = obj_out_path.with_suffix(".mtl")
    png_out_path = obj_out_path.with_suffix(".png")

    # 1) salva PNG (assicurati BGR uint8)
    cv2.imwrite(str(png_out_path), tex_img_bgr_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 2) scrivi OBJ
    tri1 = tri.copy() + 1  # OBJ 1-based
    vt = uv_coords01[:, :2].copy()  # usa solo U,V se uv_coords è (N,3)

    with open(obj_out_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_out_path.name}\n")
        f.write("usemtl material_0\n")
        for v in verts_xyz:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in vt:
            f.write(f"vt {t[0]} {t[1]}\n")
        for a, b, c in tri1:
            # usa stesso indice per v e vt (1..Nvert) — le UV sono per-vertice
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    # 3) scrivi MTL
    with open(mtl_out_path, "w", encoding="utf-8") as m:
        m.write("newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n")
        m.write(f"map_Kd {png_out_path.name}\n")

class visualize:
    def __init__(self, result_dict, args):

        self.items = ['render_shape','render_face']
        self.result_dict = result_dict
        option_list = ['ldm68', 'ldm106', 'ldm106_2d', 'ldm134', 'seg', 'seg_visible']
        for i in option_list:
            if i in self.result_dict.keys():
                self.items.append(i)
        
        self.visualize_dict = []
        self.save_dict = {}
        self.args = args

    def visualize_and_output(self, trans_params, img, save_path, img_name):
        # assert batch_size = 1
        self.visualize_dict.append(img)

        render_shape = (self.result_dict['render_shape'][0]*255).astype(np.uint8)
        render_face = (self.result_dict['render_face'][0]*255).astype(np.uint8)
        render_mask  = (np.stack((self.result_dict['render_mask'][0][:,:,0],)*3, axis=-1)*255).astype(np.uint8)

        if trans_params is not None:
            render_shape = back_resize_crop_img(render_shape, trans_params, np.zeros_like(img), resample_method = Image.BICUBIC)
            render_face = back_resize_crop_img(render_face, trans_params, np.zeros_like(img), resample_method = Image.BICUBIC)
            render_mask = back_resize_crop_img(render_mask, trans_params, np.zeros_like(img), resample_method = Image.NEAREST)

        render_shape = ((render_shape/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8)[:,:,::-1]
        self.visualize_dict.append(render_shape)
        render_face = ((render_face/255. * render_mask/255. + img[:,:,::-1]/255. * (1 - render_mask/255.))*255).astype(np.uint8)[:,:,::-1]
        self.visualize_dict.append(render_face)

        if 'ldm68' in self.items:
            ldm68=self.result_dict['ldm68'][0]
            ldm68[:, 1] = 224 -1 - ldm68[:, 1]
            if trans_params is not None:
                ldm68 = back_resize_ldms(ldm68, trans_params)
            img_ldm68 = plot_kpts(img, ldm68)
            self.visualize_dict.append(img_ldm68)
            self.save_dict['ldm68'] = ldm68

        if 'ldm106' in self.items:
            ldm106=self.result_dict['ldm106'][0]
            ldm106[:, 1] = 224 -1 - ldm106[:, 1]
            if trans_params is not None:
                ldm106 = back_resize_ldms(ldm106, trans_params)
            img_ldm106 = plot_kpts(img, ldm106)
            self.visualize_dict.append(img_ldm106)
            self.save_dict['ldm106'] = ldm106

        if 'ldm106_2d' in self.items:
            ldm106_2d=self.result_dict['ldm106_2d'][0]
            ldm106_2d[:, 1] = 224 -1 - ldm106_2d[:, 1]
            if trans_params is not None:
                ldm106_2d = back_resize_ldms(ldm106_2d, trans_params)
            img_ldm106_2d = plot_kpts(img, ldm106_2d)
            self.visualize_dict.append(img_ldm106_2d)
            self.save_dict['ldm106_2d'] = ldm106_2d

        if 'ldm134' in self.items:
            ldm134=self.result_dict['ldm134'][0]
            ldm134[:, 1] = 224 -1 - ldm134[:, 1]
            if trans_params is not None:
                ldm134 = back_resize_ldms(ldm134, trans_params)
            img_ldm134 = plot_kpts(img, ldm134)
            self.visualize_dict.append(img_ldm134)
            self.save_dict['ldm134'] = ldm134

        if 'seg_visible' in self.items:
            seg_visible = self.result_dict['seg_visible']
            new_seg_visible = np.zeros((img.shape[0],img.shape[1],8))
            for i in range(8):
                temp = np.stack((seg_visible[:,:,i],)*3, axis=-1)
                if trans_params is not None:
                    temp = back_resize_crop_img((temp).astype(np.uint8), trans_params, np.zeros_like(img), resample_method = Image.NEAREST)[:,:,::-1]
                new_seg_visible[:,:,i] = temp[:,:,0]*255

            new_seg_visible_one = np.zeros((img.shape[0],img.shape[1],1))
            for i in range(8):
                new_seg_visible_one[new_seg_visible[:,:,i]==255]=i+1
            self.visualize_dict.append(show_seg_visble(new_seg_visible_one, img))
            self.save_dict['seg_visible'] = new_seg_visible_one
            self.save_dict['seg_visible_labels'] = new_seg_visible_one.astype(np.uint8)

        if 'seg' in self.items:
            seg = self.result_dict['seg']
            new_seg = np.zeros((img.shape[0],img.shape[1],8))
            for i in range(8):
                temp = np.stack((seg[:,:,i],)*3, axis=-1)
                if trans_params is not None:
                    temp = back_resize_crop_img((temp).astype(np.uint8), trans_params, np.zeros_like(img), resample_method = Image.NEAREST)[:,:,::-1]
                new_seg[:,:,i] = temp[:,:,0]*255

                temp2 = img.copy()
                temp2[new_seg[:,:,i]==255]=np.array([200,200,100])
                self.visualize_dict.append(temp2)
            
            self.save_dict['seg'] = new_seg

        # please note that the coordinates of .obj do not account for the trans_params.
        if self.args.extractTex:
            # v3d_new = self.result_dict['v3d'][0].copy()
            # v3d_new[..., -1] = 10 - v3d_new[..., -1]
            # write_obj_with_colors(os.path.join(save_path, img_name + '_extractTex.obj'), v3d_new, self.result_dict['tri'], self.result_dict['extractTex'])
            # # cv2.imwrite(os.path.join(save_path, img_name + '_extractTex_uv.png'), (self.result_dict['extractTex_uv']*255).astype(np.uint8)[:,:,::-1])
            v3d_new = self.result_dict['v3d'][0].copy()
            v3d_new[..., -1] = 10 - v3d_new[..., -1]

            # Texture UV dall’output (float 0-1 -> uint8 BGR)
            tex_uv = (self.result_dict['extractTex_uv'] * 255.0).astype(np.uint8)
            tex_uv_bgr = tex_uv[:, :, ::-1]  # RGB->BGR per cv2.imwrite

            # UV per-vertice e triangoli (già in result_dict)
            uv_coords01 = self.result_dict['uv_coords']  # shape (Nvert, 2 o 3), range [0,1]
            tri = self.result_dict['tri']

            out_obj = os.path.join(save_path, img_name + '_extractTex.obj')
            save_obj_with_uv_texture(out_obj, v3d_new, tri, uv_coords01, tex_uv_bgr)

        # please note that the coordinates of .obj do not account for the trans_params.
        if self.args.useTex:
            v3d_new = self.result_dict['v3d'][0].copy()
            v3d_new[..., -1] = 10 - v3d_new[..., -1]
            write_obj_with_colors(os.path.join(save_path, img_name + '_pcaTex.obj'), v3d_new, self.result_dict['tri'], self.result_dict['face_texture'][0])

        len_visualize_dict = len(self.visualize_dict)
        if len(self.visualize_dict) < 4:
            img_res = np.ones((img.shape[0], len(self.visualize_dict) * img.shape[1], 3), dtype=np.uint8) * 255
        else:
            img_res = np.ones((np.ceil(len_visualize_dict/4).astype(np.int32) * img.shape[0], 4 * img.shape[1], 3), dtype=np.uint8) * 255
        for i, image in enumerate(self.visualize_dict):
            row = i // 4
            col = i % 4
            x_start = col * img.shape[1]
            y_start = row * img.shape[0]
            x_end = x_start + img.shape[1]
            y_end = y_start + img.shape[0]
            img_res[y_start:y_end, x_start:x_end] = image

        cv2.imwrite(os.path.join(save_path, img_name + '.png'), img_res)
        #np.save(os.path.join(save_path, img_name + '.npy'), self.save_dict)

        # source_tex = None
        # if self.args.extractTex:
        #     # se hai usato save_obj_with_uv_texture, avrai <img_name>_extractTex.png accanto
        #     cand = Path(save_path) / f"{img_name}_extractTex.png"
        #     if cand.exists():
        #         source_tex = str(cand)

        W, H = img.shape[1], img.shape[0]

        # trans_params "normalizzati": se w0,h0 sono scambiati, riallineali a (W,H)
        tp = np.array(trans_params, dtype=np.float32).reshape(-1)
        w0, h0, s, tx, ty = [float(x) for x in tp[:5]]
        if abs(w0 - W) + abs(h0 - H) > abs(w0 - H) + abs(h0 - W):
            w0, h0 = float(W), float(H)                  # swap se più vicino a (H,W)
        tp_fix = np.array([w0, h0, s, tx, ty], dtype=np.float32)

        _npz = {
            'trans_params': tp_fix,
            'img_size':     np.array([W, H], dtype=np.int32),
        }

        if 'seg_visible_labels' in self.save_dict:
            segvis = self.save_dict['seg_visible_labels'][:, :, 0].astype(np.uint8)  # (H,W) con valori 0..8
            _npz['seg_visible_labels_full'] = segvis

        # v2d_full (usa SEMPRE tp_fix)
        if 'v2d' in self.result_dict:
            v2d_crop = self.result_dict['v2d'][0].astype(np.float32)
            v2d_crop[:, 1] = 224 - 1 - v2d_crop[:, 1]
            v2d_full = back_resize_ldms(v2d_crop.copy(), tp_fix).astype(np.float32)
            _npz['v2d_full'] = v2d_full

        np.savez(os.path.join(save_path, f"{img_name}_projections.npz"), **_npz)

        # Se conosci il path dell’immagine originale, passalo qui (vedi nota sotto)
        #_save_trans_and_manifest(save_path, img_name, trans_params, source_tex_path=source_tex, source_image_path=None)






