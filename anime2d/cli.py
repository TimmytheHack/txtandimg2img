# anime2d/cli.py
from __future__ import annotations
from pathlib import Path
import typer
import json, shutil, subprocess

from anime2d import __version__, banner
from anime2d.utils.paths import ensure_dirs, get_paths, write_gitignore
from anime2d.utils.config import save_default_config

app = typer.Typer(add_completion=False, help="anime2d: prompt→Live2D-style anime puppet (local/FOSS)")

@app.command()
def init(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config file path to write (if missing)."),
    overwrite_config: bool = typer.Option(False, "--overwrite-config", help="Overwrite existing config file."),
):
    """
    Initialize project folders, write default config if missing, and print a summary.
    """
    p = ensure_dirs()
    write_gitignore()
    cfg_path = save_default_config(config, overwrite=overwrite_config)

    typer.echo(banner())
    typer.echo(f"Version: {__version__}\n")
    typer.echo("Folders:")
    typer.echo(f"  root        : {p.root}")
    typer.echo(f"  assets      : {p.assets}")
    typer.echo(f"  models      : {p.models}")
    typer.echo(f"  outputs     : {p.outputs}")
    typer.echo(f"  configs     : {p.configs}")
    typer.echo(f"  third_party : {p.third_party}")
    typer.echo(f"\nConfig: {cfg_path} {'(overwritten)' if overwrite_config else '(created if missing)'}")

# --- stubs for forthcoming subcommands (no-op for now) ---

@app.command()
def art(
    prompt: str = typer.Option(..., help="Character description (appearance, outfit, vibe)."),
    ref: Path = typer.Option(None, help="Optional front-view reference image."),
    cfg: Path = typer.Option(Path("configs/default.yaml"), help="Config file to use."),
    strength: float = typer.Option(0.55, min=0.1, max=0.95, help="How much to deviate from reference"),
):
    """
    Generate a front-view anime portrait/upper body (Diffusers SD1.5).
    """
    from anime2d.generate.art import generate_art
    out_path = generate_art(prompt=prompt, cfg_path=cfg, ref_image=ref, strength=strength)

    typer.echo(f"Saved: {out_path}")

@app.command()
def split(
    in_: Path = typer.Option(..., "--in", help="Input art.png from `anime2d art`"),
    out: Path = typer.Option(..., "--out", help="Output layers.psd"),
    no_matte: bool = typer.Option(False, "--no-matte", help="Skip rembg (use original RGBA)"),
):

    from anime2d.split.split import split_to_psd
    out_psd, matte = split_to_psd(in_png=in_, out_psd=out, save_matte=True, no_matte=no_matte)
    typer.echo(f"PSD : {out_psd}")
    if matte:
        typer.echo(f"Matte: {matte}")


def main():
    app()

if __name__ == "__main__":
    main()
