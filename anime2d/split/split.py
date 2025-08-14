from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
import numpy as np
from PIL import Image as PILImage
from rembg import remove
from pytoshop.user import nested_layers as nl
from pytoshop import enums  # stable location for ColorMode/ColorChannel

from psd_tools.api.psd_image import PSDImage
from psd_tools.api.layers import Group as PSDGroup, Layer as PSDLayer

SCAFFOLD: List[tuple[str, List[str]]] = [
    ("Head", []),
    ("Eyes", ["EyeL_Sclera","EyeL_Iris","EyeL_Pupil","EyeR_Sclera","EyeR_Iris","EyeR_Pupil","Eyelid_Upper","Eyelid_Lower"]),
    ("Brows", ["BrowL","BrowR"]),
    ("Mouth", ["Mouth_Upper","Mouth_Lower","Teeth","Tongue"]),
    ("Hair", ["Hair_Front","Hair_Side","Hair_Back","Accessories"]),
    ("Torso", ["Neck","Torso"]),
    ("Arms", ["ArmL","ArmR"]),
]

def _alpha_matte_safe(in_png: Path) -> PILImage:
    """Run rembg on the PNG, return RGBA PIL.Image."""
    with open(in_png, "rb") as f:
        data = f.read()
    out_bytes = remove(data)  # GPU if onnxruntime-gpu installed, else CPU
    return PILImage.open(BytesIO(out_bytes)).convert("RGBA")

def _image_layer_from_rgba(name: str, rgba: PILImage, visible: bool=True) -> nl.Image:
    """
    Build a pytoshop nl.Image from RGBA with explicit channels & bounds.
    Avoid nl.Image.from_pil to ensure compatibility with Krita/Inochi.
    """
    rgba = rgba.convert("RGBA")
    W, H = rgba.size
    arr = np.array(rgba, dtype=np.uint8)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    layer = nl.Image(
        name=name,
        visible=visible,
        color_mode=enums.ColorMode.rgb,
        top=0, left=0, right=W, bottom=H,
    )
    # RGB channels
    layer.set_channel(enums.ColorChannel.red,   r)
    layer.set_channel(enums.ColorChannel.green, g)
    layer.set_channel(enums.ColorChannel.blue,  b)

    # Alpha/transparency channel (prefer official enum; fall back to -1)
    alpha_key = getattr(getattr(enums, "SpecialChannel", None), "transparency_mask", None)
    if alpha_key is None:
        alpha_key = getattr(getattr(enums, "ChannelID", None), "transparency_mask", -1)
    try:
        layer.set_channel(alpha_key, a)
    except Exception:
        pass

    # Make sure viewer can see it
    try:
        layer.opacity = 255
    except Exception:
        pass
    try:
        layer.blend_mode = getattr(enums, "BlendMode", None).normal if hasattr(enums, "BlendMode") else None
    except Exception:
        pass

    return layer


def _empty_layer(name: str, W: int, H: int, visible: bool=False) -> nl.Image:
    rgba = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    return _image_layer_from_rgba(name, rgba, visible=visible)

def _write_psd(out_psd: Path, layers_list: list[nl.Layer], size: tuple[int, int]) -> None:
    """Pack nested layers into a PsdFile, then write it."""
    H, W = size[1], size[0]  # nested_layers_to_psd wants (H, W)
    psd = nl.nested_layers_to_psd(
        layers=layers_list,
        color_mode=enums.ColorMode.rgb,
        size=(H, W),
    )
    with open(out_psd, "wb") as f:
        psd.write(f)

def _write_psd_safe(out_psd: Path, layer_images: list[tuple[str, PILImage.Image]], size: tuple[int, int]) -> None:
    """
    Flat RGBA layers with explicit channels/bounds.
    Adds CompositePreview as the LAST layer so it's topmost in most viewers.
    """
    W, H = size

    # Composite all provided layers
    composite = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    for _, im in layer_images:
        composite.alpha_composite(im.convert("RGBA"))

    # Convert all to pytoshop layers (Background hidden)
    layers_list: List[nl.Layer] = []
    for name, im in layer_images:
        vis = (name != "Background")
        layers_list.append(_image_layer_from_rgba(name, im, visible=vis))

    # Put CompositePreview LAST so it's topmost
    layers_list.append(_image_layer_from_rgba("CompositePreview", composite, visible=True))

    _write_psd(out_psd, layers_list, size)





def build_psd_scaffold(matted_rgba: PILImage, out_psd: Path) -> Path:
    W, H = matted_rgba.size

    # Prepare flat RGBA layers (no mask metadata)
    layer_images: list[tuple[str, PILImage.Image]] = []
    layer_images.append(("Background", PILImage.new("RGBA", (W, H), (0,0,0,0))))  # hidden bg; keep as transparent
    layer_images.append(("Character", matted_rgba))

    # Live2D-style empties as fully-transparent bitmaps (Creator-safe)
    for group_name, sublayers in SCAFFOLD:
        # put the group header as an empty layer (Creator will let you regroup inside if you want)
        layer_images.append((group_name, PILImage.new("RGBA", (W, H), (0,0,0,0))))
        for name in sublayers:
            layer_images.append((name, PILImage.new("RGBA", (W, H), (0,0,0,0))))

    _write_psd_safe(out_psd, layer_images, (W, H))
    return out_psd

def split_to_psd(
    in_png: Path,
    out_psd: Path,
    save_matte: bool = True,
    no_matte: bool = False,   # ← NEW
) -> Tuple[Path, Path | None]:
    out_psd.parent.mkdir(parents=True, exist_ok=True)
    matte_png = out_psd.with_name(out_psd.stem + "_matte.png")

    if no_matte:
        rgba = PILImage.open(in_png).convert("RGBA")
    else:
        rgba = _alpha_matte_safe(in_png)
        # fallback if rembg nuked alpha
        if rgba.getextrema()[3] == (0, 0):
            raw = PILImage.open(in_png).convert("RGBA")
            arr = np.array(raw, dtype=np.uint8)
            a = np.where(arr[:, :, :3].sum(axis=2) < 750, 255, 0).astype(np.uint8)
            arr[:, :, 3] = a
            rgba = PILImage.fromarray(arr, mode="RGBA")

    if save_matte and not no_matte:
        rgba.save(matte_png)

    build_psd_scaffold(rgba, out_psd)
    return out_psd, (None if (no_matte or not save_matte) else matte_png)



def _write_psd_safe(out_psd: Path, layer_images: list[tuple[str, PILImage.Image]], size: tuple[int, int]) -> None:
    """
    Creator-safe via pytoshop: flat RGBA layers, no real masks/groups.
    """
    layers_list: List[nl.Layer] = []
    for name, im in layer_images:
        # Visible unless explicitly Background
        layers_list.append(_image_layer_from_rgba(name, im, visible=(name != "Background")))

    # If everything is fully transparent, drop in a checkerboard DEBUG layer
    if all(im.getextrema()[3] == (0, 0) for _, im in layer_images):
        W, H = size
        tile = PILImage.new("RGBA", (32, 32), (0, 0, 0, 0))
        # make a simple checker
        for y in range(32):
            for x in range(32):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    tile.putpixel((x, y), (200, 200, 200, 255))
        checker = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
        for y in range(0, H, 32):
            for x in range(0, W, 32):
                checker.alpha_composite(tile, (x, y))
        layers_list.append(_image_layer_from_rgba("DEBUG_Checker", checker, visible=True))

    _write_psd(out_psd, layers_list, size)