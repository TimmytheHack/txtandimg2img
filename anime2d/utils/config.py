# anime2d/utils/config.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import copy
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 123456,
    "sd": {
        "model": "waifu-diffusion/wd-1-5-beta3",
        "steps": 28,
        "guidance": 7.0,
        "height": 768,
        "width": 512,
        "hires_fix": True,
        "controlnets": {
            "lineart": False,
            "openpose": False,
        },
        "loras": [],
        "negative": "blurry, extra arms, side view, profile",
    },
    "upscale": {
        "impl": "realesrgan-ncnn",
        "model": "realesrgan-x4plus-anime",
    },
    "split": {
        "use_sam2": True,
        "use_anime_face_parse": True,
        "post_morphology": True,
    },
    "rig": {
        "visemes": ["AA","AE","AH","AO","EH","ER","EY","IH","IY","OW","OY","UH","UW","FV","MB"],
        "physics": {
            "hair_stiffness": 0.35,
            "hair_damping": 0.12,
            "gravity": 9.8,
        },
    },
    "tracking": {
        "osf_fps": 30,
        "smooth": 0.6,
        "map": {
            "mouth_open": "ParamMouthOpen",
            "blink_left": "ParamEyeBlinkL",
            "blink_right": "ParamEyeBlinkR",
            "brow_up": "ParamBrowUpDown",
            "head_yaw": "ParamHeadYaw",
            "head_pitch": "ParamHeadPitch",
            "head_roll": "ParamHeadRoll",
        },
    },
    "voice": {
        "tts": "voicevox:JP_Zundamon",
        "clone": "openvoice:v2",
        "loudness": -23,
    },
    "export": {
        "obs": {"spout_sender": "Inochi Session"},
        "unity": {"lipsync": "rhubarb"},
        "godot": {"udp_port": 11573},
    },
}

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user_cfg)

def save_default_config(path: Path | str, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False, allow_unicode=True)
    return path
