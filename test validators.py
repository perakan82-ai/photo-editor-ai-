import numpy as np
import pytest
from PIL import Image

import validators
from validators import ValidationError, validate_and_load


@pytest.mark.parametrize("fmt,ext", [("JPEG", "a.jpg"), ("PNG", "a.png"), ("WEBP", "a.webp")])
def test_accepts_supported(rgb_image, make_bytes, fmt, ext):
    li = validate_and_load(make_bytes(rgb_image, fmt), ext)
    assert li.format == fmt and li.image.size == rgb_image.size


def test_rejects_wrong_extension(rgb_image, make_bytes):
    with pytest.raises(ValidationError):
        validate_and_load(make_bytes(rgb_image, "JPEG"), "a.gif")


def test_rejects_fake_image():
    with pytest.raises(ValidationError):
        validate_and_load(b"not an image at all" * 100, "x.jpg")


def test_rejects_real_format_mismatch(rgb_image, make_bytes):
    with pytest.raises(ValidationError):
        validate_and_load(make_bytes(rgb_image, "BMP"), "sneaky.jpg")


def test_rejects_truncated(rgb_image, make_bytes):
    data = make_bytes(rgb_image, "PNG")
    with pytest.raises(ValidationError):
        validate_and_load(data[: len(data) // 2], "a.png")


def test_rejects_empty():
    with pytest.raises(ValidationError):
        validate_and_load(b"", "a.jpg")


def test_rejects_too_big_file(monkeypatch, rgb_image, make_bytes):
    monkeypatch.setattr(validators, "MAX_FILE_BYTES", 100)
    with pytest.raises(ValidationError):
        validate_and_load(make_bytes(rgb_image, "PNG"), "a.png")


def test_rejects_too_many_pixels(monkeypatch, rgb_image, make_bytes):
    monkeypatch.setattr(validators, "MAX_PIXELS", 1000)
    with pytest.raises(ValidationError):
        validate_and_load(make_bytes(rgb_image, "PNG"), "a.png")


def test_exif_orientation_applied(make_bytes):
    img = Image.new("RGB", (100, 50), (200, 100, 50))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    li = validate_and_load(make_bytes(img, "JPEG", exif=exif.tobytes()), "r.jpg")
    assert li.image.size == (50, 100)
    assert li.exif is not None


def test_modes_rgba_and_gray(make_bytes):
    rgba = Image.new("RGBA", (40, 40), (10, 20, 30, 128))
    gray = Image.new("L", (40, 40), 90)
    assert validate_and_load(make_bytes(rgba, "PNG"), "a.png").image.mode == "RGBA"
    assert validate_and_load(make_bytes(gray, "PNG"), "g.png").image.mode == "L"


def test_original_bytes_untouched(rgb_image, make_bytes):
    data = make_bytes(rgb_image, "JPEG")
    li = validate_and_load(data, "a.jpg")
    assert li.original_bytes == data
