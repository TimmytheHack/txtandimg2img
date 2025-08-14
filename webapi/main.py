from pathlib import Path
from typing import Optional
import torch, asyncio, json, base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import io

from anime2d.generate.art import _build_base as build_base_pipe
from anime2d.generate.art import _maybe_local

APP_ROOT = Path(__file__).resolve().parents[1]

# Prefer THIS folder directly (since your model_index.json lives here)
LOCAL_DIFFUSERS_DIR = APP_ROOT / "models" / "wd15"
HUB_MODEL_ID = "waifu-diffusion/wd-1-5-beta3"

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _has_model_index(p: Path) -> bool:
    return (p / "model_index.json").exists()

_PIPELINE: Optional[object] = None
_PIPELINE_LOCK = asyncio.Lock()

async def get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # 1) If models/wd15 has model_index.json, load it as a diffusers folder
    if _has_model_index(LOCAL_DIFFUSERS_DIR):
        sd_model_id = LOCAL_DIFFUSERS_DIR.as_posix()
        sd_local = True
    else:
        # 2) Otherwise fall back to whatever _maybe_local finds (or the hub id)
        sd_model_id, sd_local = _maybe_local(HUB_MODEL_ID, fallback_dir="wd15")

    pipe = build_base_pipe(sd_model_id, local=sd_local)
    _PIPELINE = pipe
    return _PIPELINE

_PIPELINE_TXT = None
_PIPELINE_IMG = None

async def get_pipeline_pair():
    global _PIPELINE_TXT, _PIPELINE_IMG
    if _PIPELINE_TXT is not None and _PIPELINE_IMG is not None:
        return _PIPELINE_TXT, _PIPELINE_IMG

    base = await get_pipeline()  # your existing creator that returns txt2img
    # Build img2img that shares components
    img2img = StableDiffusionImg2ImgPipeline(**base.components)
    img2img.scheduler = base.scheduler
    if torch.cuda.is_available():
        img2img.to("cuda")
    else:
        img2img.enable_model_cpu_offload()
    img2img.enable_vae_tiling()

    _PIPELINE_TXT, _PIPELINE_IMG = base, img2img
    return _PIPELINE_TXT, _PIPELINE_IMG

def _b64_to_pil(b64png: str) -> Image.Image:
    raw = base64.b64decode(b64png.split(",")[-1].encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")

app = FastAPI()

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "device": _device(),
        "using": (LOCAL_DIFFUSERS_DIR.as_posix() if _has_model_index(LOCAL_DIFFUSERS_DIR) else HUB_MODEL_ID),
        "local_dir": LOCAL_DIFFUSERS_DIR.as_posix(),
        "local_has_model_index": _has_model_index(LOCAL_DIFFUSERS_DIR),
    })

# ──────────────────────────────────────────────────────────────────────────────    
# WebSocket /ws/generate
# Receives: {prompt, steps, guidance, width, height, negative, seed, image?, strength?}
# Sends:    {"type":"preview"|"final", "image": <base64 png>}
# Cancels any in-flight generation on new message.
# ──────────────────────────────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.current_task: Optional[asyncio.Task] = None
        self.cancel_event = asyncio.Event()

    async def cancel_inflight(self):
        self.cancel_event.set()
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except Exception:
                pass
        self.cancel_event.clear()
        self.current_task = None

async def _send_progress(ws, step: int, total: int):
    # Clamp & send lightweight progress message
    step = max(0, min(step, total))
    await ws.send_text(json.dumps({"type": "progress", "step": step, "total": total}))

def _snap64(x: int) -> int:
    # keep image dims multiples of 64 for SD1.x
    return max(64, int(round(x / 64)) * 64)

async def _generate_and_send(ws, state, prompt: str, cfg: dict):
    txt2img, img2img = await get_pipeline_pair()

    steps    = int(cfg.get("steps", 24))
    guidance = float(cfg.get("guidance", 7.0))
    width    = _snap64(int(cfg.get("width", 512)))
    height   = _snap64(int(cfg.get("height", 768)))
    negative = str(cfg.get("negative", "")).strip()
    seed_val = cfg.get("seed", "") or 123456
    seed     = int(seed_val)

    init_b64 = cfg.get("image") or ""       # <— base64 PNG from client (optional)
    strength = float(cfg.get("strength", 0.55))  # how much to deviate from the init image (higher = more change)

    await ws.send_text(json.dumps({"type": "started", "total": steps}))
    gen = torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(seed)

    def _check_cancel():
        if state.cancel_event.is_set():
            raise RuntimeError("cancelled")

    loop = asyncio.get_running_loop()

    # Progress callback (new API if available)
    def _progress_emit(step_idx: int):
        asyncio.run_coroutine_threadsafe(_send_progress(ws, min(step_idx + 1, steps), steps), loop)

    try:
        if init_b64:
            # ---------- IMG2IMG ----------
            init_img = _b64_to_pil(init_b64)
            init_img = init_img.resize((width, height), Image.BICUBIC)  # SD1.x wants multiples of 64

            if "callback_on_step_end" in img2img.__call__.__code__.co_varnames:
                def on_step_end(pipe_, step, timestep, cb_kwargs):
                    _check_cancel()
                    _progress_emit(step)
                    return cb_kwargs
                result = await loop.run_in_executor(
                    None,
                    lambda: img2img(
                        prompt=prompt,
                        image=init_img,
                        strength=strength,                 # <— key knob for “how much to change”
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback_on_step_end=on_step_end,
                    )
                )
            else:
                def cb(step, timestep, latents):
                    _check_cancel()
                    _progress_emit(step)
                result = await loop.run_in_executor(
                    None,
                    lambda: img2img(
                        prompt=prompt,
                        image=init_img,
                        strength=strength,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback=cb,
                        callback_steps=1,
                    )
                )
        else:
            # ---------- TXT2IMG ----------
            if "callback_on_step_end" in txt2img.__call__.__code__.co_varnames:
                def on_step_end(pipe_, step, timestep, cb_kwargs):
                    _check_cancel(); _progress_emit(step); return cb_kwargs
                result = await loop.run_in_executor(
                    None,
                    lambda: txt2img(
                        prompt=prompt,
                        negative_prompt=negative,
                        height=height, width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback_on_step_end=on_step_end,
                    )
                )
            else:
                def cb(step, timestep, latents):
                    _check_cancel(); _progress_emit(step)
                result = await loop.run_in_executor(
                    None,
                    lambda: txt2img(
                        prompt=prompt,
                        negative_prompt=negative,
                        height=height, width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback=cb, callback_steps=1,
                    )
                )
    except asyncio.CancelledError:
        return
    except RuntimeError as e:
        if "cancelled" in str(e): return
        raise

    img = result.images[0]
    buf = BytesIO(); img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    await ws.send_text(json.dumps({
        "type": "final",
        "image": b64,
        "meta": {
            "mode": ("img2img" if init_b64 else "txt2img"),
            "steps": steps, "guidance": guidance,
            "width": width, "height": height,
            "seed": seed, "negative": negative,
            "strength": strength if init_b64 else None,
        }
    }))

@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    state = SessionState()
    await ws.send_text(json.dumps({"type": "ready"}))
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            # Control message: cancel
            if data.get("type") == "cancel":
                await state.cancel_inflight()
                await ws.send_text(json.dumps({"type": "cancelled"}))
                continue

            # Generate
            prompt = str(data.get("prompt", "")).strip()
            cfg = {
                "steps": data.get("steps"),
                "guidance": data.get("guidance"),
                "width": data.get("width"),
                "height": data.get("height"),
                "negative": data.get("negative"),
                "seed": data.get("seed"),
                # ⬇️ pass through img2img fields from the client
                "image": data.get("image"),
                "strength": data.get("strength"),
            }


            await state.cancel_inflight()
            if not prompt:
                continue

            state.current_task = asyncio.create_task(_generate_and_send(ws, state, prompt, cfg))
    except WebSocketDisconnect:
        await state.cancel_inflight()
    except Exception:
        await state.cancel_inflight()
        raise