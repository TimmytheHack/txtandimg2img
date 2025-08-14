from __future__ import annotations
from pathlib import Path
import subprocess, shutil

def realesrgan_upscale(in_png: Path, out_png: Path, model_name: str = "realesrgan-x4plus-anime", scale: int = 2) -> bool:
    """
    Returns True on success.
    """
    exe = shutil.which("realesrgan-ncnn-vulkan")
    if exe is None:
        # try local vendored path
        local = Path("third_party/realesrgan/realesrgan-ncnn-vulkan.exe")
        if local.exists():
            exe = str(local)
        else:
            return False
    cmd = [exe, "-i", str(in_png), "-o", str(out_png), "-n", model_name, "-s", str(scale)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0
