# Anime2D

**Prompt → anime portrait (local / Diffusers / Windows)**

This repo contains a minimal, local-first pipeline:

* **CLI** for one-shot generation and PSD scaffolding
* **Web API (FastAPI + WebSocket)** that generates **on Enter** (no debounce), with **progress bar** and **Stop**
* **Web app (Vite + React + Tailwind)**

> This README documents the version **without multi-image blending**. You can optionally use **one** reference image (jpg/png) from the web app (img2img) or enable **lineart ControlNet** via config (CLI).

---

## Requirements

* **Windows** + PowerShell
* **Conda env**: `anime2d` (Python **3.11**)
* **GPU**: NVIDIA + CUDA 12.1 drivers

Pinned packages (see `requirements.txt`):

* PyTorch **2.4.0 + cu121**, torchvision **0.19.0 + cu121**, torchaudio **2.4.0 + cu121**
* diffusers **0.30.0**, transformers **4.43.3**, accelerate **0.33.0**, huggingface\_hub **0.24.6**
* numpy **1.26.4**, opencv-python **4.10.0.84**, pillow **10.4.0**
* safetensors **0.4.3**, timm **1.0.19**, einops **0.8.1**
* controlnet\_aux **0.0.7** (only if you enable lineart ControlNet in config)
* typer, click, rich, pyyaml, six

Optional upscaler:

* Real-ESRGAN (ncnn) via `third_party/` if you turn it on in config

---

## Project layout

```
Anime2D/
│  README.md
│  pyproject.toml
│  requirements-lock.txt
│  requirements.txt
│  LICENSE
│
├─ anime2d/
│  │  __init__.py
│  │  main.py
│  │  cli.py                # CLI: init | art | split (lite only)
│  │
│  ├─ generate/
│  │   ├─ __init__.py
│  │   ├─ art.py            # txt2img (wd-1-5-beta3). Optional: ref image → img2img in web API; ControlNet (lineart) via CLI config
│  │   └─ upscale.py        # optional Real-ESRGAN helper (can disable in config)
│  │
│  ├─ split/
│  │   ├─ __init__.py
│  │   └─ split.py          # lite PSD scaffold; --no-matte bypasses bg removal
│  │
│  ├─ utils/
│  │   ├─ config.py
│  │   ├─ paths.py
│  │   └─ __init__.py
│  │
│  └─ data/
│      └─ visemes.json      # (unused now)
│
├─ webapi/                  # FastAPI app (WS + progress + cancel)
│  └─ main.py
│
├─ web/                     # Vite + React + Tailwind web app
│  ├─ index.html
│  ├─ src/
│  │   └─ App.tsx          # Enter-to-generate UI, progress bar, Stop, negative prompt, single-image upload
│  ├─ vite.config.ts
│  └─ package.json
│
├─ models/
│  └─ wd15/                 # **Diffusers-format** folder for wd-1-5-beta3 (contains model_index.json, unet/, vae/, tokenizer/, ...)
│
├─ outputs/
│  └─ <date>/
│      ├─ art.png
│      └─ layers.psd        # from lite split (if used)
│
├─ configs/
   └─ default.yaml

```

---

## Model setup (local, offline‑friendly)

Place the **Diffusers-format** Waifu Diffusion 1.5 beta3 in `models/wd15/` so that directory contains:

```
models/wd15/
  model_index.json
  text_encoder/
  tokenizer/
  unet/
  vae/
  scheduler/
  feature_extractor/ (optional)
```

> If you only have a single `.safetensors` file, convert once to Diffusers folder or point the **web API** at a folder that already has `model_index.json`.

**Windows cache tips**

To avoid symlink issues and unwanted downloads, you can set these in PowerShell before running:

```powershell
$env:HF_HUB_OFFLINE=1            # block network downloads
$env:HF_HUB_DISABLE_SYMLINKS=1   # avoid Windows symlink privilege errors
```

---

## Installation

```powershell
# 1) Create env
conda create -n anime2d python=3.11 -y
conda activate anime2d

# 2) Install requirements (CUDA 12.1 torch index is included inside the file)
pip install -r requirements.txt

# 3) (Optional) Install Real-ESRGAN ncnn deps if you intend to upscale
#   — put binaries/models under third_party/realesrgan or adjust your PATH
```

Initialize folders + default config:

```powershell
python -m anime2d.cli init
```

This creates folders, writes `configs/default.yaml` (if missing), and prints locations.

---

## Config (`configs/default.yaml`)

Key fields used by `anime2d.generate.art`:

```yaml
seed: 123456
sd:
  model: models/wd15          # path or HF repo id; prefer local folder
  steps: 24                   # inference steps
  guidance: 7.0               # classifier-free guidance
  width: 512
  height: 768
  negative: ""
  controlnets:
    lineart: false            # set true to enable ControlNet(Lineart) in CLI
upscale:
  impl: none                  # set to "realesrgan-ncnn" to enable x2 upscaling
  model: realesrgan-x4plus-anime
```

Notes

* ControlNet(Lineart) is **off** by default. Turn it on only if you want the CLI to use a single **reference image** for lineart conditioning.
* The web app offers **single-image img2img** (jpg/png) independently of the ControlNet setting.

---

## CLI usage

Generate art:

```powershell
anime2d art --prompt "silver-haired anime idol, detailed eyes, soft lighting" \
  --cfg configs\default.yaml
```

Optionally supply a **reference image** (used only if ControlNet lineart is enabled in config):

```powershell
anime2d art --prompt "same character, school uniform" \
  --ref assets\ref.png \
  --cfg configs\default.yaml
```

Split into a PSD scaffold (lite):

```powershell
anime2d split --in outputs\2025-08-14\art.png --out outputs\2025-08-14\layers.psd
# Skip background removal if rembg is slow/unstable
anime2d split --in outputs\2025-08-14\art.png --out outputs\2025-08-14\layers.psd --no-matte
```

Outputs are written to `outputs/<date>/`.

---

## Web API (generate on Enter)

Start the server:

```powershell
uvicorn webapi.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

### Endpoints

* `GET /health` → `{ ok, device, local_dir_exists, ... }`
* `WS  /ws/generate`
  **Send** JSON:

  ```json
  {
    "prompt": "text",
    "steps": 24,
    "guidance": 7.0,
    "width": 512,
    "height": 768,
    "negative": "",
    "seed": "123456",
    "image": "data:image/png;base64,...",   // optional single reference image (jpg/png ok)
    "strength": 0.55                          // only used if image is provided
  }
  ```

  **Receive** (sequence):

  * `{ "type": "ready" }`
  * `{ "type": "started", "total": <steps> }`
  * multiple `{ "type": "progress", "step": n, "total": <steps> }`
  * `{ "type": "final", "image": <base64 PNG>, "meta": { ... } }`

  Control message:

  * `{ "type": "cancel" }` → server cancels in-flight job and sends `{ "type": "cancelled" }`

**Img2img (single image)**

* If `image` is provided, the server switches to img2img mode (no ControlNet) and **resizes** the image to the requested width/height.
* `strength` controls how much to deviate from the image (lower ≈ closer to reference). Try `0.35–0.60`.

---

## Web app

```powershell
cd web
npm install
npm run dev
```

Open the Vite URL, then:

* Type your **prompt**, press **Enter** (or click **Generate**)
* Watch the **progress bar** (one update per diffusion step)
* **Stop** cancels the current run
* Optional: pick **one** reference image (jpg/png) and adjust **strength**
* Adjust **width/height** (multiples of 64), **steps**, **guidance**, **negative prompt**, **seed**

If you run the web app from another device, set:

```
web/.env.local → VITE_API_WS=ws://<your-ip>:8000/ws/generate
```

---

## Tuning

* **Speed**: lower `steps` (e.g. 16–24), smaller dims (e.g. 448×704). The scheduler is DPMSolverMultistep.
* **Prompt fidelity**: `guidance` 6–8. Higher can overfit and reduce style variety.
* **Seed**: fixed seed for reproducible runs; click **Random** to explore.
* **Img2img strength**: 0.35–0.55 preserves more of the reference; 0.6–0.8 allows more changes.

---

## Troubleshooting (Windows)

**HF hub symlink error**

```
[WinError 1314] The client does not possess a required privilege
```

Fix: disable hub symlinks (copy instead of link):

```powershell
$env:HF_HUB_DISABLE_SYMLINKS=1
setx HF_HUB_DISABLE_SYMLINKS 1
```

Delete any half-downloaded cache folder and retry.

**Force offline** (no downloads):

```powershell
$env:HF_HUB_OFFLINE=1
```

**Black images on Windows / torch 2.4**

* We upcast VAE (`vae.config.force_upcast = True`) and enable VAE tiling.

**WS 404 or upgrade errors**

* Install websockets support: `pip install "uvicorn[standard]" fastapi pillow`
* Ensure frontend connects to `ws://localhost:8000/ws/generate`.

**Rapid typing cancels**

* Expected: we cancel in-flight when you press **Generate** again. This is normal and won’t crash.

---

## Roadmap (later)

* Multi-image guidance (blend or IP-Adapter)
* Persistent per-session settings
* History/gallery and prompt presets
* In-browser drag crop & aspect tools

---

## License

MIT (see `LICENSE`).
