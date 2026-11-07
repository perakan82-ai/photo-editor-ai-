import io

import numpy as np
import pytest
from PIL import Image

from export_utils import (default_export_format, encode_image, output_filename,
                          realesrgan_available)
from image_engine import Adjustments, apply_adjustments
from validators import validate_and_load


def test_png_lossless_roundtrip(rgb_image):
    edited = apply_adjustments(rgb_image, Adjustments(exposure=0.7, contrast=30))
    back = Image.open(io.BytesIO(encode_image(edited, "PNG")))
    assert np.array_equal(np.asarray(back), np.asarray(edited))


def test_jpeg_resolution_and_close_to_edited(rgb_image):
    edited = apply_adjustments(rgb_image, Adjustments(exposure=0.7, saturation=40))
    back = Image.open(io.BytesIO(encode_image(edited, "JPEG", 95)))
    assert back.size == rgb_image.size and back.format == "JPEG"
    err_edited = np.abs(np.asarray(back, float) - np.asarray(edited, float)).mean()
    err_orig = np.abs(np.asarray(back, float) - np.asarray(rgb_image, float)).mean()
    assert err_edited < 3 and err_orig > err_edited * 5


def test_full_pipeline_download_is_edited_and_original_unchanged(rgb_image, make_bytes):
    """Simulates the app: upload bytes -> validate -> auto edit -> encode for download."""
    from image_engine import auto_params
    upload = make_bytes(rgb_image, "JPEG", quality=95)
    upload_copy = bytes(upload)
    li = validate_and_load(upload, "photo.jpg")
    adj, _ = auto_params(li.image, "Auto Professional")
    edited = apply_adjustments(li.image, adj)
    dl = encode_image(edited, "JPEG", exif=li.exif, icc_profile=li.icc_profile)
    assert upload == upload_copy               # upload bytes unchanged
    assert dl != upload                        # download is a different file
    back = Image.open(io.BytesIO(dl))
    assert back.size == (li.width, li.height)  # original resolution
    d_edit = np.abs(np.asarray(back, float) - np.asarray(edited, float)).mean()
    d_orig = np.abs(np.asarray(back, float) - np.asarray(li.image, float)).mean()
    assert d_edit < d_orig


def test_exif_kept_orientation_reset(make_bytes):
    img = Image.new("RGB", (100, 50), (200, 100, 50))
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "TestCam"  # Make
    li = validate_and_load(make_bytes(img, "JPEG", exif=exif.tobytes()), "r.jpg")
    dl = encode_image(li.image, "JPEG", exif=li.exif)
    out_exif = Image.open(io.BytesIO(dl)).getexif()
    assert out_exif.get(0x0112) == 1
    assert out_exif.get(0x010F) == "TestCam"


def test_rgba_to_jpeg_flattens():
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    back = Image.open(io.BytesIO(encode_image(img, "JPEG")))
    assert back.mode == "RGB" and np.asarray(back).mean() > 250


def test_rgba_png_keeps_alpha():
    img = Image.new("RGBA", (20, 20), (10, 20, 30, 77))
    back = Image.open(io.BytesIO(encode_image(img, "PNG")))
    assert back.mode == "RGBA" and np.asarray(back)[0, 0, 3] == 77


def test_webp_export(rgb_image):
    back = Image.open(io.BytesIO(encode_image(rgb_image, "WEBP")))
    assert back.format == "WEBP" and back.size == rgb_image.size


def test_bad_format(rgb_image):
    with pytest.raises(ValueError):
        encode_image(rgb_image, "GIF")


def test_helpers():
    assert default_export_format("JPEG", False) == "JPEG"
    assert default_export_format("JPEG", True) == "PNG"
    assert output_filename("my pic.final.jpeg", "PNG") == "my pic.final_edited.png"
    assert output_filename("../../evil", "JPEG").endswith("_edited.jpg")
    assert isinstance(realesrgan_available(), bool)
