# anime2d/__init__.py
__all__ = ["__version__", "banner"]

__version__ = "0.1.0"

def banner() -> str:
    return (
        "anime2d — prompt → Live2D-style puppet (local/FOSS)\n"
        "https://github.com/your-org/anime2d\n"
    )
