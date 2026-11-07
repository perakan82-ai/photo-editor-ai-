import numpy as np
import pytest
from PIL import Image

from image_engine import (MODES, PARAM_RANGES, Adjustments, analyze_image, apply_adjustments,
                          auto_params, before_after_split, luminance, to_float)


def mean_lum(img):
    return float(luminance(to_float(img.convert("RGB"))).mean())


def mean_sat(img):
    a = to_float(img.convert("RGB"))
    return float((a.max(2) - a.min(2)).mean())


def test_identity_keeps_pixels(rgb_image):
    out = apply_adjustments(rgb_image, Adjustments())
    diff = np.abs(np.asarray(out, int) - np.asarray(rgb_image, int))
    assert diff.max() <= 1


def test_original_not_modified(rgb_image):
    before = np.asarray(rgb_image).copy()
    apply_adjustments(rgb_image, Adjustments(exposure=1, contrast=50, sharpness=80, denoise=50))
    assert np.array_equal(before, np.asarray(rgb_image))


def test_size_preserved(rgb_image):
    out = apply_adjustments(rgb_image, Adjustments(exposure=0.5, sharpness=50))
    assert out.size == rgb_image.size


@pytest.mark.parametrize("name,val,check", [
    ("exposure", 1.0, lambda a, b: mean_lum(b) > mean_lum(a) + 0.05),
    ("exposure", -1.0, lambda a, b: mean_lum(b) < mean_lum(a) - 0.05),
    ("brightness", 50, lambda a, b: mean_lum(b) > mean_lum(a) + 0.03),
    ("brightness", -50, lambda a, b: mean_lum(b) < mean_lum(a) - 0.03),
    ("saturation", 60, lambda a, b: mean_sat(b) > mean_sat(a) * 1.2),
    ("saturation", -100, lambda a, b: mean_sat(b) < 0.01),
    ("vibrance", 60, lambda a, b: mean_sat(b) > mean_sat(a)),
])
def test_global_adjustments_direction(rgb_image, name, val, check):
    out = apply_adjustments(rgb_image, Adjustments(**{name: val}))
    assert check(rgb_image, out)


def test_contrast_increases_std(rgb_image):
    a = luminance(to_float(rgb_image)).std()
    b = luminance(to_float(apply_adjustments(rgb_image, Adjustments(contrast=60)))).std()
    assert b > a * 1.2


def test_temperature_warms(rgb_image):
    a = to_float(rgb_image).mean((0, 1))
    b = to_float(apply_adjustments(rgb_image, Adjustments(temperature=80))).mean((0, 1))
    assert (b[0] - b[2]) > (a[0] - a[2]) + 0.05


def test_tint_magenta_lowers_green(rgb_image):
    a = to_float(rgb_image).mean((0, 1))
    b = to_float(apply_adjustments(rgb_image, Adjustments(tint=80))).mean((0, 1))
    assert b[1] < a[1]


def test_shadows_lift_dark_areas(dark_image):
    a = luminance(to_float(dark_image))
    b = luminance(to_float(apply_adjustments(dark_image, Adjustments(shadows=80))))
    assert np.percentile(b, 10) > np.percentile(a, 10)


def test_highlights_recover_bright():
    arr = np.full((200, 200, 3), 240, np.uint8)
    arr[:, :100] = 60
    img = Image.fromarray(arr)
    out = to_float(apply_adjustments(img, Adjustments(highlights=-80)))
    assert out[:, 150:].mean() < 240 / 255 - 0.02


def test_sharpness_increases_detail(rgb_image):
    import cv2
    yy, xx = np.mgrid[0:240, 0:320]
    checker = (((xx // 20) + (yy // 20)) % 2 * 160 + 50).astype(np.uint8)
    blurred = Image.fromarray(cv2.GaussianBlur(np.dstack([checker] * 3), (0, 0), 1.5))
    g0 = cv2.Laplacian(np.asarray(blurred.convert("L")), cv2.CV_32F).var()
    g1 = cv2.Laplacian(np.asarray(apply_adjustments(blurred, Adjustments(sharpness=100)).convert("L")),
                       cv2.CV_32F).var()
    assert g1 > g0 * 1.3


def test_denoise_reduces_noise():
    rng = np.random.default_rng(1)
    clean = np.full((200, 200, 3), 128, np.float32)
    noisy = np.clip(clean + rng.normal(0, 12, clean.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy)
    out = np.asarray(apply_adjustments(img, Adjustments(denoise=80)), np.float32)
    assert out.std() < noisy.astype(np.float32).std() * 0.6


def test_rgba_alpha_preserved():
    arr = np.zeros((64, 64, 4), np.uint8)
    arr[..., :3] = 100
    arr[..., 3] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    img = Image.fromarray(arr, "RGBA")
    out = apply_adjustments(img, Adjustments(exposure=1))
    assert out.mode == "RGBA"
    assert np.array_equal(np.asarray(out)[..., 3], arr[..., 3])


def test_grayscale_supported():
    img = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (32, 1)), "L")
    out = apply_adjustments(img, Adjustments(contrast=30))
    assert out.size == img.size and out.mode == "RGB"


def test_clamping():
    adj = Adjustments(exposure=10, contrast=-500, sharpness=-5).clamped()
    assert adj.exposure == PARAM_RANGES["exposure"][1]
    assert adj.contrast == -100 and adj.sharpness == 0


def test_auto_brightens_dark_image(dark_image):
    adj, stats = auto_params(dark_image, "Auto Professional")
    assert adj.exposure > 0
    out = apply_adjustments(dark_image, adj)
    assert mean_lum(out) > mean_lum(dark_image) + 0.05
    assert stats.notes


def test_auto_corrects_colour_cast(rgb_image):
    arr = np.asarray(rgb_image).astype(np.float32)
    arr[..., 2] *= 0.6  # strong yellow/warm cast
    warm = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    adj, _ = auto_params(warm, "Auto Professional")
    assert adj.temperature < 0  # cools the image down


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_run(rgb_image, mode):
    adj, _ = auto_params(rgb_image, mode)
    out = apply_adjustments(rgb_image, adj)
    assert out.size == rgb_image.size
    if mode != "Manual Adjustment":
        assert not adj.is_identity()
    else:
        assert adj.is_identity()


def test_portrait_is_gentler_than_event(rgb_image):
    p, _ = auto_params(rgb_image, "Natural Portrait")
    e, _ = auto_params(rgb_image, "Event & Social")
    assert p.vibrance < e.vibrance and p.sharpness < e.sharpness


def test_analyze_stats_ranges(rgb_image):
    s = analyze_image(rgb_image)
    assert 0 <= s.p01 <= s.median_lum <= s.p99 <= 1


def test_before_after_split(rgb_image):
    after = Image.new("RGB", rgb_image.size, (255, 0, 0))
    comp = np.asarray(before_after_split(rgb_image, after, 0.5))
    assert tuple(comp[10, -5]) == (255, 0, 0)
    assert np.array_equal(comp[10, 5], np.asarray(rgb_image)[10, 5])
