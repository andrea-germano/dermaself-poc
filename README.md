# Pipeline di Anonimizzazione Facciale 3D (basata su 3DDFA‑V3)

Questa repo esegue una pipeline **end‑to‑end** per anonimizzare volti partendo da immagini 2D:
1. **Face 3D** – ricostruzione del volto in 3D e proiezioni ausiliarie (3DDFA‑V3).
2. **Segmentation** – generazione di una maschera RGBA che rimuove occhi (e opz. bocca) dall’immagine originale.
3. **Anonymization** – *morph* della mesh verso una faccia media + re‑bake della texture ad alta risoluzione.
4. **Renderer** – render 2D dell’oggetto 3D anonimo (pytorch3d).

> Nota: non è richiesto **nvdiffrast**. La ricostruzione usa il renderer **CPU** fornito da 3DDFA‑V3 (Cython).

---

## Requisiti & Setup

### 1) Clona e crea l’ambiente
```bash
git clone https://github.com/andrea-germano/dermaself-poc.git
cd dermaself-poc

# Consigliato: conda (Windows/Linux/macOS)
conda create -n DDFAV3-anon python=3.8 -y
conda activate DDFAV3-anon

# PyTorch (versione testata)
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
# In alternativa (Windows/Linux specifica): vedi tabella su pytorch.org
```

### 2) Dipendenze Python
```bash
pip install -r requirements.txt
```

### 3) Compila il renderer **CPU** (Cython)
```bash
cd ddfa_v3/util/cython_renderer/
python setup.py build_ext -i
cd ../../..
```
> Windows: se appare l’errore *“Microsoft Visual C++ 14.x is required”*, installa i **Build Tools** di Visual Studio (componenti C++ & Windows SDK), riapri il terminale e ripeti la compilazione.

### 4) (Opzionale ma utile) Renderer 2D con **pytorch3d**
Serve solo per lo **Step 4 – Renderer** (se vuoi un PNG 2D). Se ti basta il **.obj** anonimo puoi saltare.
```bash
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
> Linux: assicurati di avere i tool di compilazione (es. `build-essential`).

### 5) Asset e pesi
Scarica i file richiesti da 3DDFA‑V3 nella cartella `ddfa_v3/assets/` (modelli, indici landmark, ecc.).
- Per lo **Step 3 – Anonymization** serve **`face_model.npy` oppure `.npz`** con la chiave **`'u'`** (coordinate mean face).
- Vedi `ddfa_v3/assets/README.md` per i link ai pesi.

---

## Struttura dati consigliata
```
dermaself-poc/
├─ input/                 # immagini di input (.png/.jpg) - nomi univoci
├─ out/
│  ├─ face3d/            # step 1 – output per immagine in sottocartelle
│  ├─ seg/               # step 2 – PNG RGBA con occhi/bocca rimossi
│  ├─ anon/              # step 3 – OBJ + texture anonima
│  └─ render/            # step 4 – PNG renderizzato
└─ ddfa_v3/              # codice 3DDFA‑V3 + assets/
```

---

## Esecuzione passo‑passo

### Step 1 — Face 3D (ricostruzione con 3DDFA‑V3)
Script: `face_3d.py`

Esempio (CPU):
```bash
python face_3d.py \
  --inputpath input/ \
  --savepath out/face3d \
  --iscrop 1 \
  --detector retinaface \
  --ldm68 1 --ldm106 1 --ldm106_2d 1 --ldm134 1 \
  --seg_visible 1 --seg 1 \
  --useTex 0 --extractTex 1 \
  --backbone resnet50
```
- **Input**: cartella con immagini (`--inputpath`).  
- **Output**: in `--savepath` viene creata **una sottocartella per immagine** (in base al nome file) con:
  - `*_extractTex.obj` – mesh triangolata con UV estratta dall’immagine;
  - `*_projections.npz` – proiezioni/segmentazioni pre‑calcolate. Servono i campi:
    - `seg_visible_labels_full` (per Step 2);
    - `v2d_full` (per Step 3).
- **Detector**: `retinaface` (default) o `mtcnn`. Con immagini già allineate 224×224, puoi usare `--iscrop 0`.

> La modalità con CUDA offerta dal modello 3ddfav3 non è supportata al momento, il rendering viene fatto unicamente su cpu

---

### Step 2 — Segmentation (creazione maschera RGBA)
Script: `segmentation.py`

Esempio:
```bash
python segmentation.py \
  --in_root out/face3d \
  --images_root input \
  --out_root out/seg \
  --remove_mouth 1 \
  --margin_px 6 \
  --feather_px 1.5
```
- **Input**: `--in_root` = cartella dello Step 1 con le **sottocartelle** per immagine.  
  Per ogni sottocartella si legge `<stem>_projections.npz` e la chiave `seg_visible_labels_full`.
- **images_root**: cartella con le **stesse immagini originali**, non ricorsiva (match su `<stem>.png|jpg|jpeg|bmp`).  
  Le dimensioni devono coincidere con quelle usate nelle proiezioni.
- **Output**: un PNG RGBA per immagine in `--out_root`, nome `"<stem>_masked.png"`:  
  - RGB con **inpaint** delle zone rimosse;
  - Alpha=0 nelle zone rimosse; **feather** controlla la morbidezza dei bordi.

Opzioni chiave:
- `--remove_mouth 1` rimuove anche la bocca (etichette 6,7). Metti `0` per lasciare la bocca.
- `--margin_px` aggiunge dilatazione (px) attorno a occhi/bocca.
- `--feather_px` = sigma Gauss per transizioni morbide (0 disattiva).

---

### Step 3 — Anonymization (morph + re‑bake texture)
Script: `anonymization.py`

Esempio:
```bash
python anonymization.py \
  --in_root out/face3d \
  --images_root input \
  --face_model ddfa_v3/assets/face_model.npy \
  --out_root out/anon \
  --alpha 1.0 \
  --tex_size 2048 \
  --keep_pose
```
- **Input** (per ogni sottocartella in `--in_root`):
  - `*_extractTex.obj` (mesh con UV);
  - `<stem>_projections.npz` con **`v2d_full`** (proiezione 2D per vertice);
  - immagine originale corrispondente da `--images_root`.
- **Output** per immagine in `--out_root/<sottocartella>`:
  - `<stem>.obj` – **mesh anonima** (morph verso mean face);
  - `<stem)_texture.png` – **texture UV ricostruita** ad alta risoluzione.
- `--face_model` deve contenere la chiave `'u'` (mean face, 3N valori).  
- `--alpha` (0..1) controlla l’intensità del morph (1.0 = mean face pieno).  
- `--keep_pose` preserva la posa globale (Umeyama); togli l’opzione per normalizzare anche la posa.
- `--tex_size` (es. 1024, 2048, 4096) imposta la risoluzione della texture UV.

> Lo script **si ferma** al primo errore (file mancanti, shape mismatch, ecc.) così da segnalare subito i prerequisiti mancanti.

---

### Step 4 — Renderer (PNG dal modello anonimo)
Script: `renderer.py` (usa **pytorch3d**).

1) Apri `renderer.py` e imposta:
```python
OBJ_PATH = "out/anon/<stem>/<stem>.obj"  # path ad un .obj dello step 3
OUT_PATH = "out/render/<stem>_front.png" # dove salvare l’immagine
IMG_SIZE = 1024                          # lato in pixel
MARGIN   = 0.95                          # quanta parte dell’inquadratura riempire
BG_WHITE = False                         # True => sfondo bianco, False => PNG RGBA
```
2) Lancia:
```bash
python renderer.py
```
- La camera è **ortografica frontale** con auto‑fit.
- L’output è un **PNG RGBA** (alpha binario > 0.5), pronto per compositing.

> Se non vuoi installare pytorch3d, salta questo step: lo **.obj** e la **texture** dello Step 3 sono già utilizzabili in DCC/engine esterni.

---

## Suggerimenti & Troubleshooting

- **`ModuleNotFoundError: Cython`**: installa `cython` nell’ambiente e ricompila lo step 3 (`setup.py build_ext -i`).
- **Compilazione C/C++ su Windows**: installa i *Microsoft C++ Build Tools* (MSVC 14.x) e il *Windows 10/11 SDK*.
- **`FileNotFoundError: ./assets/face_model.npy`**: aggiorna i path per cercare in `ddfa_v3/assets/`.
- **TorchVision warning `pretrained` deprecato**: usa `weights=...` se modifichi i backbone (non blocca l’esecuzione).
- **`*_projections.npz` mancante**: ripeti lo Step 1 con `--seg_visible 1 --extractTex 1`.
- **Mismatch dimensioni immagine/segmentazione**: Step 2 richiede la **stessa risoluzione** dell’immagine usata nello Step 1 (stesso `<stem>`).

---

## Licenze e crediti
- Ricostruzione 3D basata su **3DDFA‑V3** (accademico). Questa repo integra componenti per l’anonimizzazione e tool di rendering.
- Verifica le licenze dei pacchetti terzi (PyTorch, OpenCV, pytorch3d, ecc.) prima di uso commerciale.
