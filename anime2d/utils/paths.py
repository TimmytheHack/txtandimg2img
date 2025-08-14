# anime2d/utils/paths.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

PROJECT_DEFAULTS = ("assets", "models", "outputs", "configs", "third_party")

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    assets: Path
    models: Path
    outputs: Path
    configs: Path
    third_party: Path

def project_root() -> Path:
    # Assume current working dir is the project root (where pyproject.toml lives)
    return Path.cwd()

def get_paths() -> ProjectPaths:
    r = project_root()
    return ProjectPaths(
        root=r,
        assets=r / "assets",
        models=r / "models",
        outputs=r / "outputs",
        configs=r / "configs",
        third_party=r / "third_party",
    )

def ensure_dirs() -> ProjectPaths:
    p = get_paths()
    for d in (p.assets, p.models, p.outputs, p.configs, p.third_party):
        d.mkdir(parents=True, exist_ok=True)
    return p

def dated_output_dir(date_str: str | None = None) -> Path:
    """
    Returns outputs/YYYY-MM-DD (does not create it).
    """
    from datetime import datetime
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return get_paths().outputs / date_str

def write_gitignore():
    """
    Create a minimal .gitignore if it doesn't exist.
    """
    path = project_root() / ".gitignore"
    if path.exists():
        return
    content = "\n".join([
        "__pycache__/",
        ".pytest_cache/",
        ".venv/",
        ".hf/",
        "outputs/",
        "models/",
        "third_party/",
        "*.egg-info/",
        "dist/",
        "build/",
    ]) + "\n"
    path.write_text(content, encoding="utf-8")
