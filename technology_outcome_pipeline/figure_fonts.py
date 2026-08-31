"""Portable TrueType fonts for deterministic, non-generative figure rendering."""

from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from PIL import ImageFont

_fonts: dict[bool, str] = {}


def configure_fonts(regular: Path | None = None, bold: Path | None = None) -> None:
    for is_bold, override in ((False, regular), (True, bold)):
        if override is not None and not override.is_file():
            raise ValueError(f"Font file does not exist: {override}")
        _fonts[is_bold] = str(override) if override else findfont(
            FontProperties(family="DejaVu Sans", weight="bold" if is_bold else "normal")
        )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if not _fonts:
        configure_fonts()
    return ImageFont.truetype(_fonts[bold], size)


def font_files() -> dict[str, str]:
    if not _fonts:
        configure_fonts()
    return {"regular": _fonts[False], "bold": _fonts[True]}
