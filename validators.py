"""
validators.py
Safe loading and validation of uploaded images.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 25 * 1024 * 1024        # 25 MB
MAX_PIXELS = 40_000_000                  # 40 MP
MAX_SIDE = 10_000
MIN_SIDE = 16

# Guard against decompression bombs
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2


class ValidationError(Exception):
    """Raised with a user-friendly (Thai) message."""


@dataclass
class LoadedImage:
    image: Image.Image          # EXIF-orientation-corrected working copy
    format: str                 # JPEG / PNG / WEBP
    original_mode: str
    width: int
    height: int
    file_size: int
    exif: bytes | None          # raw EXIF (orientation reset later on export)
    icc_profile: bytes | None
    original_bytes: bytes       # untouched upload


def validate_and_load(data: bytes, filename: str = "") -> LoadedImage:
    if not data:
        raise ValidationError("ไฟล์ว่างเปล่า")
    if len(data) > MAX_FILE_BYTES:
        raise ValidationError(f"ไฟล์ใหญ่เกินไป (สูงสุด {MAX_FILE_BYTES // (1024*1024)} MB)")

    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError("รองรับเฉพาะไฟล์ JPG, JPEG, PNG, WEBP")

    # 1) Integrity check (verify() invalidates the object, so reopen afterwards)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ValidationError("ไฟล์ไม่ใช่รูปภาพ หรือไฟล์เสียหาย") from e
    except Image.DecompressionBombError as e:
        raise ValidationError("รูปมีขนาดพิกเซลใหญ่เกินกำหนด") from e

    # 2) Real format (not just the extension)
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"ชนิดไฟล์จริงคือ {fmt} ซึ่งไม่รองรับ")

    # 3) Full decode
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as e:
        raise ValidationError("รูปมีขนาดพิกเซลใหญ่เกินกำหนด") from e
    except Exception as e:  # truncated files etc.
        raise ValidationError("ไม่สามารถอ่านรูปได้ (ไฟล์อาจไม่สมบูรณ์)") from e

    w, h = img.size
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ValidationError("รูปเล็กเกินไป")
    if w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
        raise ValidationError(f"รูปใหญ่เกินไป (สูงสุด {MAX_PIXELS // 1_000_000} ล้านพิกเซล, ด้านละไม่เกิน {MAX_SIDE}px)")

    if img.mode not in ("RGB", "RGBA", "L", "LA", "P", "CMYK", "I;16", "I", "PA"):
        raise ValidationError(f"โหมดสี {img.mode} ไม่รองรับ")

    exif = img.info.get("exif")
    icc = img.info.get("icc_profile")
    original_mode = img.mode

    # 4) Normalise: apply EXIF orientation, convert odd modes safely
    work = ImageOps.exif_transpose(img.copy())
    if work.mode in ("I;16", "I"):
        work = work.point(lambda v: v / 256).convert("L")
    if work.mode == "CMYK":
        work = work.convert("RGB")
        icc = None  # CMYK profile no longer applies
    if work.mode == "PA":
        work = work.convert("RGBA")

    return LoadedImage(
        image=work, format=fmt, original_mode=original_mode,
        width=work.width, height=work.height, file_size=len(data),
        exif=exif, icc_profile=icc, original_bytes=data,
    )
