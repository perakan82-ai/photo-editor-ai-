"""
export_utils.py
Encode the edited image for download at original resolution.
"""

from __future__ import annotations

import importlib.util
import io

from PIL import Image

JPEG_QUALITY = 95
EXPORT_FORMATS = ["JPEG", "PNG", "WEBP"]
MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def _clean_exif(exif_bytes: bytes | None) -> bytes | None:
    """Keep camera metadata but reset Orientation to 1 because pixels are
    already rotated upright. Drops the embedded thumbnail (it would show the
    unedited photo)."""
    if not exif_bytes:
        return None
    try:
        exif = Image.Exif()
        exif.load(exif_bytes)
        exif[0x0112] = 1  # Orientation
        return exif.tobytes()
    except Exception:
        return None


def default_export_format(source_format: str, has_alpha: bool) -> str:
    if has_alpha:
        return "PNG"
    return source_format if source_format in EXPORT_FORMATS else "JPEG"


def encode_image(img: Image.Image, fmt: str = "JPEG", quality: int = JPEG_QUALITY,
                 exif: bytes | None = None, icc_profile: bytes | None = None) -> bytes:
    """Return encoded bytes. Pixel dimensions are never changed here."""
    fmt = fmt.upper()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {fmt}")
    quality = int(min(max(quality, 70), 100))

    out = img
    params: dict = {}
    if icc_profile:
        params["icc_profile"] = icc_profile
    clean = _clean_exif(exif)
    if clean:
        params["exif"] = clean

    if fmt == "JPEG":
        if out.mode in ("RGBA", "LA", "P"):
            # JPEG has no transparency: flatten on white
            rgba = out.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            out = bg
        elif out.mode != "RGB":
            out = out.convert("RGB")
        params.update(quality=quality, subsampling=0 if quality >= 90 else 2, optimize=True)
    elif fmt == "PNG":
        params.update(optimize=False, compress_level=6)
    elif fmt == "WEBP":
        params.update(quality=quality, method=4)

    buf = io.BytesIO()
    out.save(buf, format=fmt, **params)
    return buf.getvalue()


def output_filename(original_name: str, fmt: str, suffix: str = "_edited") -> str:
    base = original_name.rsplit(".", 1)[0] if original_name else "photo"
    base = "".join(c for c in base if c.isalnum() or c in "-_ .").strip(" .") or "photo"
    return f"{base}{suffix}.{EXT[fmt.upper()]}"


# ---------------------------------------------------------------------------
# AI upscale (future phase) - only reports real availability, never fakes it
# ---------------------------------------------------------------------------

def realesrgan_available() -> bool:
    """True only if the Real-ESRGAN Python package is actually installed.
    The upscale feature itself is not implemented in this MVP."""
    return importlib.util.find_spec("realesrgan") is not None
