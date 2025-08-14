from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionImg2ImgPipeline,
)
from controlnet_aux import LineartDetector
from anime2d.utils.config import load_config
from anime2d.utils.paths import dated_output_dir, get_paths
from anime2d.generate.upscale import realesrgan_upscale

def _build_img2img_from_base(base_pipe):
    """Build an img2img pipeline reusing the same components as the base txt2img pipe."""
    img2img = StableDiffusionImg2ImgPipeline(**base_pipe.components)
    img2img.scheduler = base_pipe.scheduler
    if torch.cuda.is_available():
        img2img.to("cuda")
    else:
        img2img.enable_model_cpu_offload()
    img2img.enable_vae_tiling()
    return img2img

def _maybe_local(model_id: str, fallback_dir: Optional[str] = None) -> tuple[str, bool]:
    p = Path(model_id)
    if p.exists():
        return str(p), True
    if fallback_dir:
        local = get_paths().models / fallback_dir
        if local.exists():
            return str(local), True
    return model_id, False

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _build_base(sd_model_id: str, local: bool) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id, torch_dtype=torch.float16, safety_checker=None, local_files_only=local
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    dev = _device()
    if dev == "cuda":
        pipe.to(dev)
    else:
        pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.vae.config.force_upcast = True  # crucial for Windows/torch2.4 black image issues
    return pipe

def _build_controlnet(lineart_model_id: str, local: bool) -> ControlNetModel:
    return ControlNetModel.from_pretrained(lineart_model_id, torch_dtype=torch.float16, local_files_only=local)

def _build_cnet_pipe(sd_model_id: str, sd_local: bool, cnet: ControlNetModel) -> StableDiffusionControlNetPipeline:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=cnet, torch_dtype=torch.float16, safety_checker=None, local_files_only=sd_local
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    dev = _device()
    if dev == "cuda":
        pipe.to(dev)
    else:
        pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.vae.config.force_upcast = True
    return pipe

def generate_art(prompt: str,
                 out_path: str,
                 *,
                 ref_image: str | None = None,
                 strength: float = 0.55,
                 width: int = 512,
                 height: int = 768,
                 steps: int = 24,
                 guidance: float = 7.0,
                 negative: str = "",
                 seed: int | None = 123456,
                 **kwargs):

    # 1) Build your usual txt2img base pipe (you already have this in _build_base)
    pipe = _build_base(...)   # <- whatever you already do to load wd15 locally
    gen  = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    # 2) Snap dims to multiples of 64
    def _snap64(x: int) -> int: return max(64, (x // 64) * 64)
    W, H = _snap64(width), _snap64(height)

    # 3) If there’s a reference image, switch to IMG2IMG
    if ref_image:
        # Load JPG/PNG; PIL handles both
        init_img = Image.open(ref_image).convert("RGB").resize((W, H), Image.BICUBIC)

        img2img = _build_img2img_from_base(pipe)

        result = img2img(
            prompt=prompt,
            image=init_img,
            strength=float(strength),        # lower = closer to the image (0.2–0.45), higher = more change (0.6–0.8)
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=gen,
        )
    else:
        # Pure TXT2IMG
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=H, width=W,
            generator=gen,
        )

    image = result.images[0]
    image.save(out_path)
    return out_path
