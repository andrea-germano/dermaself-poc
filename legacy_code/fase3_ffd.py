import json, shutil
from pathlib import Path
import argparse
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------- I/O OBJ+MTL ----------------
def save_obj_with_uv_texture(obj_out_path, verts_xyz, faces, uv_coords01, tex_src_png_path, flip_v=False):
    obj_out_path = Path(obj_out_path); obj_out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_dst_png_path = obj_out_path.with_suffix(".png")
    tex_src_png_path = Path(tex_src_png_path)
    if not tex_src_png_path.exists(): raise FileNotFoundError(f"PNG non trovata: {tex_src_png_path}")
    if tex_src_png_path.resolve()!=tex_dst_png_path.resolve(): shutil.copyfile(tex_src_png_path, tex_dst_png_path)
    mtl_out_path = obj_out_path.with_suffix(".mtl")
    vt = uv_coords01[:, :2].copy()
    if flip_v: vt[:,1] = 1.0 - vt[:,1]
    with open(obj_out_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_out_path.name}\nusemtl material_0\n")
        for v in verts_xyz: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in vt:       f.write(f"vt {t[0]} {t[1]}\n")
        tri1 = faces.astype(np.int64)+1
        for a,b,c in tri1: f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    with open(mtl_out_path, "w", encoding="utf-8") as m:
        m.write("newmtl material_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n")
        m.write(f"map_Kd {tex_dst_png_path.name}\n")

def _rewrite_mtl_to_texture(obj_path, new_png_path):
    mtl_path = Path(obj_path).with_suffix(".mtl")
    if not mtl_path.exists(): return
    lines = mtl_path.read_text(encoding="utf-8").splitlines()
    new_lines=[]
    for ln in lines:
        if ln.strip().lower().startswith("map_kd "): new_lines.append(f"map_Kd {Path(new_png_path).name}")
        else: new_lines.append(ln)
    mtl_path.write_text("\n".join(new_lines), encoding="utf-8")

# --------------- LOAD BUNDLE ------------------
def load_bundle(stem_dir: Path):
    mesh_dir = stem_dir/"mesh"; masks_dir = stem_dir/"masks_uv"
    manifest = json.loads((stem_dir/"manifest.json").read_text(encoding="utf-8"))
    V  = np.load(mesh_dir/"verts.npy")       # (N,3)
    F  = np.load(mesh_dir/"faces.npy")       # (F,3)
    UV = np.load(mesh_dir/"uv_coords.npy")   # (N,2)
    U  = np.load(mesh_dir/"u_mean.npy")      # (N,3)
    pro = json.loads((mesh_dir/"procrustes.json").read_text(encoding="utf-8"))
    s, R, t = pro["s"], np.asarray(pro["R"], np.float64), np.asarray(pro["t"], np.float64)
    U_aligned = (s*(U.astype(np.float64)@R.T) + t).astype(np.float32)
    masks8 = np.load(masks_dir/"masks_uv_coarse8.npy")  # HxWx8
    tex_src = Path(manifest.get("source_tex", "")) if "source_tex" in manifest else None
    if not tex_src or not tex_src.exists():
        guess = stem_dir/f"{stem_dir.name}.png"; tex_src = guess if guess.exists() else None
    trans_params = np.load(mesh_dir/"trans_params.npy")
    return V, F, UV, U_aligned, tex_src, masks8, trans_params, manifest

# --------------- GEOMETRIA --------------------
def cotmatrix(V, F):
    V = V.astype(np.float64); F = F.astype(np.int64)
    i0,i1,i2 = F[:,0],F[:,1],F[:,2]
    v0,v1,v2 = V[i0],V[i1],V[i2]
    e0 = v1-v2; e1 = v2-v0; e2 = v0-v1
    def _cot(a,b):
        num = (a*b).sum(axis=1); den = np.linalg.norm(np.cross(a,b),axis=1)+1e-12
        return num/den
    cot0=_cot(e1,e2); cot1=_cot(e2,e0); cot2=_cot(e0,e1)
    I=np.r_[i1,i2,i0,i2,i0,i1]; J=np.r_[i2,i1,i2,i0,i1,i0]
    W=0.5*np.r_[cot0,cot0,cot1,cot1,cot2,cot2]
    n=V.shape[0]; W=sp.coo_matrix((W,(I,J)),shape=(n,n)).tocsr()
    d=np.array(W.sum(axis=1)).ravel()
    return sp.diags(d)-W

def arap_solve_soft(V, F, target_pos, w_data=70.0, iters=10):
    V0 = V.astype(np.float64).copy(); n=V.shape[0]
    L = cotmatrix(V0,F); A=(L+sp.diags(np.full(n,w_data))).tocsr()
    X=V0.copy()
    for _ in range(max(1,iters)):
        B=w_data*target_pos
        for d in range(3): X[:,d]=spla.spsolve(A,B[:,d])
    return X.astype(np.float32)

def _edge_list(F):
    E=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); E=np.unique(np.sort(E,axis=1),axis=0); return E

def strain_guard(V0,V1,F,max_stretch=1.6, iters=6):
    E=_edge_list(F)
    for _ in range(max(1,iters)):
        e0=V0[E[:,0]]-V0[E[:,1]]
        e1=V1[E[:,0]]-V1[E[:,1]]
        L0=np.linalg.norm(e0,axis=1)+1e-12; L1=np.linalg.norm(e1,axis=1)+1e-12
        rmax=float(np.max(L1/L0))
        if rmax<=float(max_stretch): return V1
        s=float(max_stretch)/rmax
        V1 = V0 + (V1 - V0)*s
    return V1

# ------ LIMITA ANISOTROPIA (evita faccia oblonga) ------
def limit_anisotropy(V0, V1, max_y_vs_xz=1.08):
    """Limita lo scale globale lungo Y rispetto alla media di X/Z."""
    b0 = V0.max(0) - V0.min(0) + 1e-9
    b1 = V1.max(0) - V1.min(0) + 1e-9
    sx, sy, sz = b1[0]/b0[0], b1[1]/b0[1], b1[2]/b0[2]
    avg_xz = 0.5*(sx+sz)
    limit = avg_xz*float(max_y_vs_xz)
    if sy <= limit: return V1
    # riduci solo la componente Y intorno al mid-y
    midy = 0.5*(V0.max(0)[1] + V0.min(0)[1])
    scale = limit/sy
    V1c = V1.copy()
    V1c[:,1] = midy + (V1c[:,1]-midy)*scale
    return V1c

# --------- REGIONI (UV con V-flip + fallback) ----------
def region_vertices(UV, masks8, uv_vflip=True, verbose=True):
    H,W,C=masks8.shape; assert C>=8
    u_px=np.clip((UV[:,0]*(W-1)).astype(int),0,W-1)
    v01 = 1.0-UV[:,1] if uv_vflip else UV[:,1]
    v_px=np.clip((v01*(H-1)).astype(int),0,H-1)
    px=np.c_[u_px,v_px]
    def m(k): return masks8[px[:,1],px[:,0],k]>0
    I={}
    I['eye_r']=np.where(m(0)|m(2))[0]; I['eye_l']=np.where(m(1)|m(3))[0]
    I['nose']=np.where(m(4))[0]; I['mouth']=np.where(m(5)|m(6))[0]
    # canale 7 = pelle; le regioni sotto sono HEURISTICHE → le rendiamo opzionali via pesi
    skin=masks8[px[:,1],px[:,0],7]>0; u=UV[:,0]; v=v01
    I['cheek_r']=np.where(skin & (u>0.55) & (v<0.65))[0]
    I['cheek_l']=np.where(skin & (u<0.45) & (v<0.65))[0]
    I['temple_r']=np.where(skin & (u>0.60) & (v<0.35))[0]
    I['temple_l']=np.where(skin & (u<0.40) & (v<0.35))[0]
    I['chin']=np.where(skin & (v>0.70))[0]
    I={k:v for k,v in I.items() if v.size>0}
    if verbose:
        print("  [classes] " + (", ".join([f"{k}:{len(v)}" for k,v in I.items()]) if I else "NONE"))
    covered=sum(len(v) for v in I.values())
    if covered<0.02*len(UV):
        print("  [warn] poche classi → fallback UV boxes")
        I_fb={}
        I_fb['eye_r']=np.where((u>0.58)&(u<0.85)&(v>0.20)&(v<0.45))[0]
        I_fb['eye_l']=np.where((u<0.42)&(u>0.15)&(v>0.20)&(v<0.45))[0]
        I_fb['nose' ]=np.where((u>0.40)&(u<0.60)&(v>0.40)&(v<0.65))[0]
        I_fb['mouth']=np.where((u>0.35)&(u<0.65)&(v>0.65)&(v<0.85))[0]
        I_fb={k:v for k,v in I_fb.items() if v.size>0}
        print("  [fallback] " + (", ".join([f"{k}:{len(v)}" for k,v in I_fb.items()]) if I_fb else "NONE"))
        if I_fb: return I_fb
    return I

def region_centers(V, idx_dict):
    return {k: V[idx].mean(axis=0).astype(np.float64) for k,idx in idx_dict.items()}

# --------- CALIBRAZIONE mm→unità ----------
def estimate_mm2u(V, UV, masks8):
    idx = region_vertices(UV, masks8, uv_vflip=True, verbose=False)
    mm2u = None
    if 'eye_r' in idx and 'eye_l' in idx and idx['eye_r'].size and idx['eye_l'].size:
        cr = V[idx['eye_r']].mean(axis=0); cl = V[idx['eye_l']].mean(axis=0)
        dist_u = float(np.linalg.norm(cr - cl))
        if dist_u > 1e-9: mm2u = dist_u / 62.0  # 62 mm inter-oculare
    if mm2u is None:
        width_u = float(np.max(V[:,0]) - np.min(V[:,0]))
        mm2u = width_u / 160.0  # 160 mm larghezza faccia
    mm2u = max(mm2u, 1e-6)
    print(f"  [calib] 1 mm ≈ {mm2u:.6f} unità;  10 mm ≈ {10*mm2u:.6f}")
    return mm2u

# --------- FFD ------------------------------
def _bspline_weights(u):
    u = np.asarray(u, np.float64)
    B0 = (1 - u)**3 / 6.0
    B1 = (3*u**3 - 6*u**2 + 4) / 6.0
    B2 = (-3*u**3 + 3*u**2 + 3*u + 1) / 6.0
    B3 = (u**3) / 6.0
    return np.stack([B0, B1, B2, B3], axis=1)  # (N,4)

def prep_ffd_lattice(V, grid=(9,11,9), pad=0.03):
    Vd=V.astype(np.float64); vmin=Vd.min(axis=0); vmax=Vd.max(axis=0)
    size=vmax-vmin; vmin=vmin-pad*size; vmax=vmax+pad*size; size=vmax-vmin
    gx,gy,gz=map(int,grid)
    P=(Vd-vmin)/(size+1e-12); P=np.clip(P,0.0,1.0-1e-9)
    def prep_axis(p,g):
        s=p*(g-1); i=np.floor(s).astype(np.int32); u=(s-i).astype(np.float64); i0=i-1
        idx=np.stack([np.clip(i0+k,0,g-1) for k in range(4)],axis=1)
        wts=_bspline_weights(u)
        return idx,wts
    ix,wx=prep_axis(P[:,0],gx); iy,wy=prep_axis(P[:,1],gy); iz,wz=prep_axis(P[:,2],gz)
    grid_coords=np.stack(np.meshgrid(np.linspace(0,1,gx),np.linspace(0,1,gy),np.linspace(0,1,gz),indexing='ij'),axis=-1).reshape(-1,3)
    cp_pos=vmin+grid_coords*size
    return {'vmin':vmin,'size':size,'grid':(gx,gy,gz),'ix':ix,'iy':iy,'iz':iz,'wx':wx,'wy':wy,'wz':wz,'cp_pos':cp_pos}

def apply_ffd(V, pre, cp_offsets_u):
    gx,gy,gz=pre['grid']; G=gx*gy*gz
    off=cp_offsets_u.reshape(G,3).astype(np.float64)
    ix,iy,iz=pre['ix'],pre['iy'],pre['iz']; wx,wy,wz=pre['wx'],pre['wy'],pre['wz']
    N=V.shape[0]; disp=np.zeros((N,3),np.float64)
    for a in range(4):
        for b in range(4):
            wab=(wx[:,a]*wy[:,b])[:,None]
            ia=ix[:,a]; jb=iy[:,b]
            for c in range(4):
                w=wab*wz[:,c][:,None]; kc=iz[:,c]; lin=(ia*gy+jb)*gz+kc
                disp+=off[lin]*w
    return (V.astype(np.float64)+disp).astype(np.float32)

def gauss_weight(cp_pos, center, sigma_u):
    d2=np.sum((cp_pos-center[None,:])**2,axis=1); return np.exp(-d2/(2.0*sigma_u*sigma_u))

# --------- OFFSETS (overdrive + pesi) ----------
def build_offsets(V, UV, masks8, pre, mm2u, strength=1.2, class_var=0.7, asym=0.3,
                  w_eyes=1.0, w_nose=1.0, w_mouth=1.0, w_temples=0.3, w_cheeks=0.1, w_chin=0.1, w_face=0.0,
                  seed=0, symmetry=True):
    rng=np.random.default_rng(seed)
    cp=pre['cp_pos']; gx,gy,gz=pre['grid']
    off=np.zeros((gx*gy*gz,3),np.float64)
    idx=region_vertices(UV, masks8, uv_vflip=True, verbose=True)
    C=region_centers(V,idx)
    midx=float(V[:,0].mean())

    # overdrive curve
    s=float(np.clip(strength,0.0,1.5))
    s1=min(s,1.0); s2=max(0.0,s-1.0)
    gain = s1 + 0.65*(s2**1.2)

    def sminmax(a,b): return a+(b-a)*gain
    base = {
        'eye_sep'  : sminmax(4, 12)*w_eyes,
        'cheek'    : sminmax(3, 11)*w_cheeks,
        'temple'   : sminmax(3, 10)*w_temples,
        'jaw_in'   : sminmax(3, 11)*w_chin,
        'nose_nar' : sminmax(2,  9)*w_nose,
        'nose_out' : sminmax(0,  6)*w_nose,
        'mouth_dn' : sminmax(1,  4)*w_mouth,
        'face_tall': sminmax(2,  8)*w_face,
        'sigma_mm' : sminmax(22, 28)
    }
    def vary(v): return v*(1.0 + class_var*rng.uniform(-1,1))
    eye_sep,cheek_out,temple_out= vary(base['eye_sep']), vary(base['cheek']), vary(base['temple'])
    jaw_in,nose_narrow,nose_out  = vary(base['jaw_in']), vary(base['nose_nar']), vary(base['nose_out'])
    mouth_down,face_tall,sigma_mm= vary(base['mouth_dn']), vary(base['face_tall']), base['sigma_mm']

    # shifts posizionali (mm), scalati dai pesi della classe
    eye_v = class_var*3.0 * rng.choice([-1,1]) * w_eyes
    eye_h = class_var*2.0 * rng.choice([-1,1]) * w_eyes
    nose_v= class_var*2.5 * rng.choice([-1,1]) * w_nose
    nose_h= class_var*1.5 * rng.choice([-1,1]) * w_nose
    mouth_v=class_var*3.0 * rng.choice([-1,1]) * w_mouth

    sigma_u = sigma_mm * mm2u

    def push(center3d, vec3, amp_mm):
        if center3d is None or amp_mm==0: return
        w=gauss_weight(cp, np.asarray(center3d,np.float64), sigma_u)[:,None]
        off[:] += w * (np.asarray(vec3,np.float64)[None,:] * (amp_mm * mm2u))

    # Occhi (3ddfav3): forte
    if 'eye_r' in C and 'eye_l' in C:
        push(C['eye_r'], (+1,0,0), +eye_sep/2); push(C['eye_l'], (-1,0,0), +eye_sep/2)
        for k in ['eye_r','eye_l']:
            if k in C:
                push(C[k], (0, np.sign(eye_v), 0), abs(eye_v))
                push(C[k], (np.sign(eye_h), 0, 0), abs(eye_h))
    # Naso (3ddfav3): forte
    if 'nose' in C:
        dir_mid=np.array([midx-C['nose'][0],0,0],np.float64); n=np.linalg.norm(dir_mid)
        if n>1e-9: dir_mid/=n
        push(C['nose'], dir_mid, +nose_narrow)
        push(C['nose'], (0, np.sign(nose_v), 0), abs(nose_v))
        push(C['nose'], (np.sign(nose_h), 0, 0), abs(nose_h))
        push(C['nose'], (0,0, np.sign(rng.uniform(-1,1))), +nose_out)
    # Bocca (3ddfav3): forte
    if 'mouth' in C:
        push(C['mouth'], (0,-1,0), +mouth_down)
        push(C['mouth'], (0, np.sign(mouth_v), 0), abs(mouth_v))

    # Euristiche (deboli/azzerabili coi pesi)
    if 'cheek_r' in C and cheek_out>0: push(C['cheek_r'], (+1,0,0), +cheek_out)
    if 'cheek_l' in C and cheek_out>0: push(C['cheek_l'], (-1,0,0), +cheek_out)
    if 'temple_r' in C and temple_out>0: push(C['temple_r'], (+1,0,0), +temple_out)
    if 'temple_l' in C and temple_out>0: push(C['temple_l'], (-1,0,0), +temple_out)
    if 'chin' in C and jaw_in>0:
        dir_mid=np.array([midx-C['chin'][0],0,0],np.float64); n=np.linalg.norm(dir_mid)
        if n>1e-9: dir_mid/=n; push(C['chin'], dir_mid, +jaw_in)

    # Volto più alto (spesso causa "oblungo"): regolato da w_face
    if face_tall>0:
        vmin,size=pre['vmin'],pre['size']; cp_norm=(cp-vmin[None,:])/(size[None,:]+1e-12)
        off[:,1]+=np.clip((cp_norm[:,1]-0.5)*2.0,0,1)*(face_tall*mm2u)

    # Rumore low-freq (asimmetrico controllato)
    rnd_amp_mm = (0.5 + 1.2*gain) * 3.0
    if rnd_amp_mm>0:
        gx,gy,gz=pre['grid']
        rnd = rng.standard_normal((gx,gy,gz,3)) * (rnd_amp_mm*mm2u)
        for _ in range(2):
            rnd[1:-1,:,:]=(rnd[0:-2,:,:]+rnd[1:-1,:,:]+rnd[2:,:,:])/3.0
            rnd[:,1:-1,:]=(rnd[:,0:-2,:]+rnd[:,1:-1,:]+rnd[:,2:,:])/3.0
            rnd[:,:,1:-1]=(rnd[:,:,0:-2]+rnd[:,:,1:-1]+rnd[:,:,2:])/3.0
        off += rnd.reshape(-1,3)

    # Simmetria + asimmetria
    cpL=cp[:,0]<midx; cpR=~cpL
    from scipy.spatial import cKDTree
    treeR=cKDTree(cp[cpR]); mirL=np.c_[2*midx-cp[cpL][:,0], cp[cpL][:,1], cp[cpL][:,2]]
    _,j=treeR.query(mirL, k=1)
    symR=off[cpL].copy(); symR[:,0]*=-1
    offR=off[cpR]; offR[j]=0.5*offR[j]+0.5*symR; off_sym=off.copy(); off_sym[cpR]=offR

    if asym>0:
        rng2=np.random.default_rng(seed+1337)
        jitter=(rng2.standard_normal(off.shape)*(1.4*mm2u*gain))
        xc=(cp[:,0]-midx); fall=np.clip(1.0 - np.abs(xc)/(np.abs(xc).max()+1e-9), 0,1)[:,None]
        jitter*= (0.2 + 0.8*fall)
        off = (1.0-asym)*off_sym + asym*(off_sym + jitter)
    else:
        off = off_sym

    # Anchor al collo
    vmin,size=pre['vmin'],pre['size']; cp_norm=(cp-vmin[None,:])/(size[None,:]+1e-12)
    low = cp_norm[:,1] < 0.15
    off[low]*=0.25
    return off

# -------------- FFD DRIVER -------------------
def ffd_anonymize(V, F, UV, masks8, strength=1.2, class_var=0.7, asym=0.3, grid=(9,11,9), seed=0,
                  w_eyes=1.0, w_nose=1.0, w_mouth=1.0, w_temples=0.3, w_cheeks=0.1, w_chin=0.1, w_face=0.0):
    mm2u = estimate_mm2u(V, UV, masks8)
    pre=prep_ffd_lattice(V, grid=grid, pad=0.03)
    off=build_offsets(V, UV, masks8, pre, mm2u,
                      strength=strength, class_var=class_var, asym=asym, seed=seed,
                      w_eyes=w_eyes, w_nose=w_nose, w_mouth=w_mouth,
                      w_temples=w_temples, w_cheeks=w_cheeks, w_chin=w_chin, w_face=w_face)
    V_ffd=apply_ffd(V, pre, off)

    # Clip (in mm → unità)
    s=float(np.clip(strength,0,1.5)); gain = min(s,1.0) + 0.65*max(0.0,s-1.0)**1.2
    clip_mm = 10 + 12*gain
    m = clip_mm * mm2u
    d=V_ffd-V; nrm=np.linalg.norm(d,axis=1,keepdims=True)+1e-12
    V_ffd = V + d*np.minimum(1.0, m/nrm)

    # Limita anisotropia Y vs X/Z (anti "alieno")
    V_ffd = limit_anisotropy(V, V_ffd, max_y_vs_xz=1.08)

    # Strain-guard + ARAP polish
    guard = 1.65 - 0.15*gain
    V_ffd = strain_guard(V, V_ffd, F, max_stretch=guard, iters=6)
    V_out = arap_solve_soft(V_ffd, F, target_pos=V_ffd, w_data=70.0, iters=10)

    disp=np.linalg.norm(V_out - V, axis=1)
    print(f"  [apply] disp mediana: {np.median(disp):.6f}, 95p: {np.percentile(disp,95):.6f}, max: {disp.max():.6f} (unità)")
    return V_out

# -------------- RE-BAKE (sempre da mesh originale) --------------
def _project_to_crop224(V_obj):
    V_cam=V_obj.astype(np.float32).copy(); V_cam[:,2]=10.0-V_cam[:,2]
    f,cx,cy=1015.0,112.0,112.0; P=np.array([[f,0,cx],[0,f,cy],[0,0,1]],np.float32)
    proj=V_cam@P.T; return proj[:,:2]/proj[:,2:3]

def _back_resize_ldms(ldms, trans_params):
    w0,h0,s,tx,ty=trans_params; T=224.0
    w=int(w0*s); h=int(h0*s)
    left=int(w/2-T/2+(tx-w0/2)*s); up=int(h/2-T/2+(h0/2-ty)*s)
    out=ldms.astype(np.float32).copy()
    out[:,0]=(out[:,0]+left)/float(w)*float(w0); out[:,1]=(out[:,1]+up)/float(h)*float(h0)
    return out

def _uv_to_pixels(UV,R): return np.c_[UV[:,0]*(R-1),(1.0-UV[:,1])*(R-1)].astype(np.float32)

def unsharp(img_bgr, amount=0.35, radius=1.2):
    blur=cv2.GaussianBlur(img_bgr,(0,0),radius); return np.clip(cv2.addWeighted(img_bgr,1+amount,blur,-amount,0),0,255).astype(np.uint8)

def rebake_uv_png_roi_fast(stem_dir, V_src, F, UV, out_png_path,
                           uv_res=3072, downsample=2, sharpen=True, min_uv_area_px=3.0, threads=0):
    stem_dir=Path(stem_dir); mesh_dir=stem_dir/"mesh"
    manifest=json.loads((stem_dir/"manifest.json").read_text(encoding="utf-8"))
    try:
        if threads and threads>0: cv2.setNumThreads(int(threads))
    except Exception: pass
    src_img_path = Path(manifest.get("source_image", ""))
    if not src_img_path.exists(): print("  [re-bake] source_image mancante"); return None
    trans_params=np.load(mesh_dir/"trans_params.npy")

    R=int(max(512,uv_res))
    atlas=np.zeros((R,R,3),np.uint8)
    mask_full=np.zeros((R,R),np.uint8)
    uv_px=_uv_to_pixels(UV,R)

    v2d=_project_to_crop224(V_src).astype(np.float32); v2d[:,1]=223.0-v2d[:,1]
    v2d=_back_resize_ldms(v2d,trans_params).astype(np.float32)

    src=cv2.imread(str(src_img_path),cv2.IMREAD_COLOR)
    if src is None: raise FileNotFoundError(f"Immagine non leggibile: {src_img_path}")

    tri=F.astype(np.int64); min_area=float(min_uv_area_px)
    for a,b,c in tri:
        dst=np.float32([uv_px[a],uv_px[b],uv_px[c]])
        area=0.5*abs((dst[1,0]-dst[0,0])*(dst[2,1]-dst[0,1])-(dst[2,0]-dst[0,0])*(dst[1,1]-dst[0,1]))
        if area<min_area: continue
        src_tri=np.float32([v2d[a],v2d[b],v2d[c]])
        x0=int(np.floor(np.min(dst[:,0]))); x1=int(np.ceil(np.max(dst[:,0])))+1
        y0=int(np.floor(np.min(dst[:,1]))); y1=int(np.ceil(np.max(dst[:,1])))+1
        if x1<=0 or y1<=0 or x0>=R or y0>=R: continue
        x0=max(0,x0); y0=max(0,y0); x1=min(R,x1); y1=min(R,y1); bw=x1-x0; bh=y1-y0
        if bw<=1 or bh<=1: continue
        dst_roi=dst.copy(); dst_roi[:,0]-=x0; dst_roi[:,1]-=y0
        mask=np.zeros((bh,bw),np.uint8); cv2.fillConvexPoly(mask,np.int32(dst_roi+0.5),255)
        M=cv2.getAffineTransform(src_tri,dst_roi)
        patch=cv2.warpAffine(src,M,(bw,bh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        m3=(mask.astype(np.float32)/255.0)[:,:,None]
        atlas[y0:y1,x0:x1]=(patch*m3+atlas[y0:y1,x0:x1]*(1.0-m3)).astype(np.uint8)
        mask_full[y0:y1, x0:x1] |= mask

    outR=R//max(1,int(downsample))
    atlas=cv2.resize(atlas,(outR,outR),interpolation=cv2.INTER_AREA)
    mask_small=cv2.resize(mask_full,(outR,outR),interpolation=cv2.INTER_NEAREST)

    # Anti-bleeding sui bordi UV
    kernel=np.ones((3,3),np.uint8)
    border=cv2.morphologyEx(mask_small, cv2.MORPH_GRADIENT, kernel)
    atlas_dil=cv2.dilate(atlas, kernel, iterations=1)
    atlas[border>0]=atlas_dil[border>0]

    if sharpen: atlas=unsharp(atlas,amount=0.35,radius=1.2)
    out_png_path=Path(out_png_path); out_png_path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out_png_path), atlas, [cv2.IMWRITE_PNG_COMPRESSION,0])
    print(f"  [re-bake] PNG: {out_png_path}"); return str(out_png_path)

# -------------- PIPELINE ---------------------
def process_all(bundles_dir="stage3_bundles", out_dir="stage3_morphed",
                strength=1.2, class_var=0.7, asym=0.3, grid="9x11x9", seed=0, uv_res=3072, flip_v=False,
                w_eyes=1.0, w_nose=1.0, w_mouth=1.0, w_temples=0.3, w_cheeks=0.1, w_chin=0.1, w_face=0.0):
    bundles_dir=Path(bundles_dir); out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stems=[p for p in bundles_dir.iterdir() if p.is_dir()]
    if not stems: print(f"Nessun bundle in {bundles_dir}"); return
    for stem_dir in stems:
        stem=stem_dir.name; print(f"[{stem}] Carico…")
        V,F,UV,U_aligned,tex_src,masks8,trans_params,manifest = load_bundle(stem_dir)
        if tex_src is None or not Path(tex_src).exists(): print("  ! Texture PNG assente, salto"); continue

        print("  Deformazione FFD (focus 3ddfav3)…")
        gx,gy,gz = map(int, grid.lower().split('x'))
        V_new = ffd_anonymize(
            V, F, UV, masks8,
            strength=float(strength), class_var=float(class_var), asym=float(asym),
            grid=(gx,gy,gz), seed=int(seed),
            w_eyes=float(w_eyes), w_nose=float(w_nose), w_mouth=float(w_mouth),
            w_temples=float(w_temples), w_cheeks=float(w_cheeks),
            w_chin=float(w_chin), w_face=float(w_face),
        )

        out_mesh_dir = out_dir/stem/"mesh"; out_mesh_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_mesh_dir/"verts_morphed.npy", V_new.astype(np.float32))
        out_obj = out_dir/stem/f"{stem}_morphed.obj"
        save_obj_with_uv_texture(out_obj, V_new, F, UV, tex_src, flip_v=bool(flip_v))
        print(f"  -> OBJ salvato: {out_obj}")

        png_out = out_obj.with_name(out_obj.stem + "_rebaked.png")
        # IMPORTANTE: campiona dalla mesh ORIGINALE V
        rebaked = rebake_uv_png_roi_fast(stem_dir, V, F, UV, png_out,
                                         uv_res=int(uv_res), downsample=2,
                                         sharpen=True, min_uv_area_px=3.0, threads=0)
        if rebaked:
            _rewrite_mtl_to_texture(out_obj, rebaked)
            print(f"  -> MTL aggiornato: {Path(rebaked).name}")

# -------------- CLI --------------------------
def main():
    ap=argparse.ArgumentParser(description="Anonimizzazione 3D Focus (FFD overdrive + pesi classe + anti-oblungo + UV re-bake)")
    ap.add_argument("--bundles_dir", default="stage3_bundles")
    ap.add_argument("--out_dir", default="stage3_morphed")
    ap.add_argument("--strength", type=float, default=1.2, help="0..1.5 (overdrive >1)")
    ap.add_argument("--class_var", type=float, default=0.7, help="0..1 variazione per classe")
    ap.add_argument("--asym", type=float, default=0.3, help="0..1 asimmetria controllata")
    ap.add_argument("--grid", default="9x11x9", help="griglia FFD es. 8x10x8 / 9x11x9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uv_res", type=int, default=3072)
    ap.add_argument("--flip_v", type=int, default=0)
    # pesi per classe (1.0=forte; 0.0=disattiva)
    ap.add_argument("--w_eyes", type=float, default=1.0)
    ap.add_argument("--w_nose", type=float, default=1.0)
    ap.add_argument("--w_mouth", type=float, default=1.0)
    ap.add_argument("--w_temples", type=float, default=0.30)
    ap.add_argument("--w_cheeks", type=float, default=0.10)
    ap.add_argument("--w_chin", type=float, default=0.10)
    ap.add_argument("--w_face", type=float, default=0.00, help="altezza volto (0=off)")
    args=ap.parse_args()
    process_all(
        bundles_dir=args.bundles_dir, out_dir=args.out_dir,
        strength=max(0.0,min(1.5,args.strength)),
        class_var=max(0.0,min(1.0,args.class_var)),
        asym=max(0.0,min(1.0,args.asym)),
        grid=args.grid, seed=args.seed, uv_res=args.uv_res, flip_v=bool(args.flip_v),
        w_eyes=args.w_eyes, w_nose=args.w_nose, w_mouth=args.w_mouth,
        w_temples=args.w_temples, w_cheeks=args.w_cheeks, w_chin=args.w_chin, w_face=args.w_face
    )

if __name__=="__main__":
    main()
