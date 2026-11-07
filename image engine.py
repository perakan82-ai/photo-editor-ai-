"""
image_engine.py
Core image-processing engine for Professional Photo Editor AI.

All processing works on a float32 RGB copy in the 0..1 range.
The input PIL image is never modified.

Two parts:
1. apply_adjustments()  - applies the real pixel operations.
2. analyze_image() / auto_params() - the "auto pro" analysis that inspects
   the photo (exposure, contrast, white balance, colour, noise, sharpness)
   and chooses natural-looking settings for the selected mode.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

MODES = ["Auto Professional", "Natural Portrait", "Event & Social", "Manual Adjustment"]

# (min, max, default) for each slider. Exposure is in EV stops.
PARAM_RANGES: Dict[str, tuple] = {
    "exposure": (-2.0, 2.0, 0.0),
    "brightness": (-100, 100, 0),
    "contrast": (-100, 100, 0),
    "highlights": (-100, 100, 0),
    "shadows": (-100, 100, 0),
    "temperature": (-100, 100, 0),
    "tint": (-100, 100, 0),
    "vibrance": (-100, 100, 0),
    "saturation": (-100, 100, 0),
    "sharpness": (0, 100, 0),
    "denoise": (0, 100, 0),
}


@dataclass
class Adjustments:
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    sharpness: float = 0.0
    denoise: float = 0.0

    def clamped(self) -> "Adjustments":
        values = {}
        for name, value in asdict(self).items():
            lo, hi, _ = PARAM_RANGES[name]
            values[name] = float(np.clip(value, lo, hi))
        return Adjustments(**values)

    def is_identity(self) -> bool:
        return all(abs(v) < 1e-9 for v in asdict(self).values())

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Rec. 709 luminance weights
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------
# Conversion helpers
# --------------------------------------------------------------------------

def split_alpha(img: Image.Image):
    """Return (rgb_image, alpha_or_None). Works for RGB, RGBA, L, LA, P, CMYK..."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return img.convert("RGB"), None


def to_float(img_rgb: Image.Image) -> np.ndarray:
    return np.asarray(img_rgb, dtype=np.float32) / 255.0


def to_pil(arr: np.ndarray) -> Image.Image:
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")


def luminance(arr: np.ndarray) -> np.ndarray:
    return arr @ _LUMA


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# Individual operations (each takes and returns float32 RGB 0..1)
# --------------------------------------------------------------------------

def adjust_exposure(arr, ev):
    if ev == 0:
        return arr
    return arr * (2.0 ** ev)


def adjust_brightness(arr, amount):
    """Gamma-style brightness: lifts/lowers mid-tones, keeps black & white points."""
    if amount == 0:
        return arr
    gamma = 2.0 ** (-amount / 100.0)  # +100 -> gamma 0.5 (brighter)
    return np.power(np.clip(arr, 0, 1), gamma)


def adjust_contrast(arr, amount):
    """S-curve-free linear contrast around mid grey 0.5."""
    if amount == 0:
        return arr
    factor = 1.0 + amount / 100.0 if amount < 0 else 1.0 + amount / 100.0 * 1.5
    return (arr - 0.5) * factor + 0.5


def adjust_highlights_shadows(arr, highlights, shadows):
    """Luminance-masked tone recovery. Negative highlights pull bright areas down,
    positive shadows open up dark areas. Applied as a luminance ratio so colour
    is preserved."""
    if highlights == 0 and shadows == 0:
        return arr
    arr = np.clip(arr, 0, 1)
    lum = luminance(arr)
    # Blur the mask a bit so local detail does not get flattened (halo-safe size)
    h, w = lum.shape
    k = max(3, int(round(min(h, w) / 200)) | 1)
    lum_s = cv2.GaussianBlur(lum, (k, k), 0)

    new_lum = lum.copy()
    if highlights != 0:
        mask = _smoothstep(0.5, 1.0, lum_s)
        amt = highlights / 100.0
        if amt < 0:  # compress toward 0.5
            new_lum = new_lum + mask * amt * 0.5 * (new_lum - 0.5) * 2 * 0.6
        else:        # brighten highlights
            new_lum = new_lum + mask * amt * (1.0 - new_lum) * 0.5
    if shadows != 0:
        mask = 1.0 - _smoothstep(0.0, 0.5, lum_s)
        amt = shadows / 100.0
        if amt > 0:  # lift shadows
            new_lum = new_lum + mask * amt * (0.5 - new_lum).clip(0) * 0.8
        else:        # deepen shadows
            new_lum = new_lum + mask * amt * new_lum * 0.6

    ratio = (new_lum + 1e-4) / (lum + 1e-4)
    return arr * ratio[..., None]


def adjust_temperature_tint(arr, temperature, tint):
    """White balance via channel gains. +temperature = warmer, +tint = magenta."""
    if temperature == 0 and tint == 0:
        return arr
    t = temperature / 100.0 * 0.25
    g = tint / 100.0 * 0.20
    gains = np.array([1.0 + t, 1.0 - g, 1.0 - t], dtype=np.float32)
    # Normalise so overall luminance stays roughly the same
    gains = gains / float(gains @ _LUMA)
    return arr * gains


def adjust_saturation(arr, amount):
    if amount == 0:
        return arr
    lum = luminance(arr)[..., None]
    factor = 1.0 + amount / 100.0
    return lum + (arr - lum) * factor


def adjust_vibrance(arr, amount):
    """Boosts muted colours more than already-saturated ones and protects
    skin-like (orange) hues so portraits stay natural."""
    if amount == 0:
        return arr
    arr_c = np.clip(arr, 0, 1)
    mx = arr_c.max(axis=2)
    mn = arr_c.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    hsv = cv2.cvtColor(arr_c.astype(np.float32), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]  # 0..360
    skin = np.exp(-((hue - 25.0) / 20.0) ** 2)  # ~orange skin tones
    weight = (1.0 - sat) * (1.0 - 0.6 * skin)
    factor = 1.0 + (amount / 100.0) * weight
    lum = luminance(arr)[..., None]
    return lum + (arr - lum) * factor[..., None]


def apply_denoise(arr, amount):
    """Edge-preserving colour denoise (OpenCV non-local means).
    Strength is kept conservative to avoid plastic skin."""
    if amount <= 0:
        return arr
    img8 = np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)
    h_l = 1.0 + amount / 100.0 * 9.0     # luminance strength 1..10
    h_c = h_l * 1.2                       # chroma noise is usually stronger
    den = cv2.fastNlMeansDenoisingColored(img8, None, h_l, h_c, 5, 11)
    # Blend back a little detail so texture survives
    blend = 0.85
    out = den.astype(np.float32) * blend + img8.astype(np.float32) * (1 - blend)
    return out / 255.0


def apply_sharpness(arr, amount):
    """Unsharp mask on luminance only (no colour fringing), radius scaled to image size."""
    if amount <= 0:
        return arr
    arr = np.clip(arr, 0, 1)
    lum = luminance(arr)
    h, w = lum.shape
    sigma = max(0.8, min(h, w) / 1500.0)
    blur = cv2.GaussianBlur(lum, (0, 0), sigma)
    detail = lum - blur
    # Threshold tiny differences so noise is not amplified
    detail = detail * _smoothstep(0.0015, 0.006, np.abs(detail))
    strength = amount / 100.0 * 1.6
    new_lum = lum + detail * strength
    ratio = (np.clip(new_lum, 0, 1) + 1e-4) / (lum + 1e-4)
    out = arr * ratio[..., None]
    return out


def _soft_clip(arr):
    """Gently roll off values above 1 instead of hard clipping."""
    if float(arr.max()) <= 1.0:
        return np.clip(arr, 0, 1)  # nothing overflows -> leave pixels untouched
    knee = 0.92
    over = arr > knee
    out = arr.copy()
    x = arr[over] - knee
    out[over] = knee + (1 - knee) * (1 - np.exp(-x / (1 - knee)))
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def apply_adjustments(img: Image.Image, adj: Adjustments) -> Image.Image:
    """Return a NEW edited PIL image with the same size. Input is not modified.
    Alpha channel (if any) is preserved."""
    adj = adj.clamped()
    rgb, alpha = split_alpha(img.copy())
    arr = to_float(rgb).copy()

    # Order matters: clean -> white balance -> tone -> colour -> detail
    arr = apply_denoise(arr, adj.denoise)
    arr = adjust_temperature_tint(arr, adj.temperature, adj.tint)
    arr = adjust_exposure(arr, adj.exposure)
    arr = adjust_highlights_shadows(arr, adj.highlights, adj.shadows)
    arr = _soft_clip(arr)
    arr = adjust_contrast(arr, adj.contrast)
    arr = adjust_brightness(np.clip(arr, 0, 1), adj.brightness)
    arr = adjust_vibrance(arr, adj.vibrance)
    arr = adjust_saturation(arr, adj.saturation)
    arr = apply_sharpness(arr, adj.sharpness)

    out = to_pil(np.clip(arr, 0, 1))
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


# --------------------------------------------------------------------------
# Auto analysis ("AI auto pro")
# --------------------------------------------------------------------------

@dataclass
class ImageStats:
    median_lum: float
    mean_lum: float
    p01: float
    p05: float
    p95: float
    p99: float
    clipped_high: float
    clipped_low: float
    mean_sat: float
    cast_rb: float   # >0 means image is too warm (red > blue)
    cast_g: float    # >0 means image is too green
    noise_sigma: float
    sharpness_score: float
    notes: list = field(default_factory=list)


def _analysis_copy(img: Image.Image, max_side: int = 1024) -> np.ndarray:
    rgb, _ = split_alpha(img)
    rgb = rgb.copy()
    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return to_float(rgb)


def analyze_image(img: Image.Image) -> ImageStats:
    arr = _analysis_copy(img)
    lum = luminance(arr)
    p01, p05, p50, p95, p99 = np.percentile(lum, [1, 5, 50, 95, 99])

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)

    # White balance: grey-world on near-neutral mid-tones only, so large
    # coloured areas (blue sky, green grass) do not fool the estimate.
    neutral = (lum > 0.12) & (lum < 0.9) & (sat < 0.22)
    frac = float(neutral.mean())
    if neutral.sum() < 200:
        neutral = (lum > 0.12) & (lum < 0.9)
        frac = 0.0
    wgt = (1.0 - sat[neutral] / 0.22).clip(0.05) if frac > 0 else np.ones(int(neutral.sum()))
    r, g, b = (float(np.average(arr[..., i][neutral], weights=wgt)) for i in range(3))
    avg = (r + g + b) / 3 + 1e-6
    confidence = float(np.clip(frac / 0.05, 0.25, 1.0))  # few neutral pixels -> weaker correction
    cast_rb = float((r - b) / avg) * confidence
    cast_g = float((g - (r + b) / 2) / avg) * confidence

    # Noise estimate on a full-resolution centre crop (downscaling hides noise)
    rgb_full, _ = split_alpha(img)
    W, H = rgb_full.size
    c = 512
    crop = rgb_full.crop((max(0, W // 2 - c), max(0, H // 2 - c),
                          min(W, W // 2 + c), min(H, H // 2 + c)))
    crop_l = luminance(to_float(crop)) * 255.0
    resid = crop_l - cv2.GaussianBlur(crop_l, (3, 3), 0)
    noise_sigma = float(np.median(np.abs(resid)) * 1.4826)

    gray8 = (lum * 255).astype(np.float32)
    lap = cv2.Laplacian(gray8, cv2.CV_32F)
    sharpness_score = float(lap.var())

    stats = ImageStats(
        median_lum=float(p50), mean_lum=float(lum.mean()),
        p01=float(p01), p05=float(p05), p95=float(p95), p99=float(p99),
        clipped_high=float((lum > 0.98).mean()), clipped_low=float((lum < 0.02).mean()),
        mean_sat=float(sat.mean()), cast_rb=cast_rb, cast_g=cast_g,
        noise_sigma=noise_sigma, sharpness_score=sharpness_score,
    )
    return stats


# Per-mode personality: target brightness and how strong each correction is.
_MODE_PROFILE = {
    "Auto Professional": dict(target=0.46, contrast=1.0, vib=1.0, sat=0.0, sharp=1.0, wb=0.6, den=1.0),
    "Natural Portrait":  dict(target=0.50, contrast=0.6, vib=0.6, sat=-0.1, sharp=0.6, wb=0.7, den=0.8),
    "Event & Social":    dict(target=0.48, contrast=1.3, vib=1.4, sat=0.15, sharp=1.2, wb=0.5, den=1.1),
}


def auto_params(img: Image.Image, mode: str = "Auto Professional") -> tuple[Adjustments, ImageStats]:
    """Analyse the photo and return natural-looking adjustments for the mode."""
    if mode not in _MODE_PROFILE:
        return Adjustments(), analyze_image(img)
    prof = _MODE_PROFILE[mode]
    s = analyze_image(img)
    notes = []

    # Exposure: bring median brightness toward target, conservatively
    med = max(s.median_lum, 0.01)
    ev = float(np.log2(prof["target"] / med)) * 0.6
    ev = float(np.clip(ev, -1.0, 1.2))
    if abs(ev) < 0.08:
        ev = 0.0
    if ev > 0:
        notes.append("ภาพค่อนข้างมืด → เพิ่ม exposure")
    elif ev < 0:
        notes.append("ภาพค่อนข้างสว่าง → ลด exposure")

    # Highlights / shadows recovery
    highlights = 0.0
    if s.p99 > 0.93 or s.clipped_high > 0.01:
        highlights = -float(np.clip((s.p99 - 0.85) * 300 + s.clipped_high * 500, 15, 60))
        notes.append("ไฮไลต์จ้า → กู้รายละเอียดส่วนสว่าง")
    elif ev > 0:
        highlights = -15.0
    shadows = 0.0
    if s.p05 < 0.08:
        shadows = float(np.clip((0.08 - s.p05) * 500, 10, 45))
        notes.append("เงามืด → เปิดรายละเอียดในเงา")

    # Contrast: flat images get more, already contrasty images get less
    tonal_range = s.p95 - s.p05
    contrast = float(np.clip((0.70 - tonal_range) * 60, -10, 25)) * prof["contrast"]
    contrast = max(contrast, 5.0 * prof["contrast"])
    if tonal_range < 0.55:
        notes.append("คอนทราสต์แบน → เพิ่มมิติ")

    # White balance: partial correction keeps the mood of the light
    temperature = float(np.clip(-s.cast_rb * 200 * prof["wb"], -20, 20))
    tint = float(np.clip(s.cast_g * 250 * prof["wb"], -12, 12))
    if abs(temperature) > 5 or abs(tint) > 5:
        notes.append("สมดุลสีขาวเพี้ยน → แก้ white balance")

    # Colour
    vibrance = float(np.clip((0.35 - s.mean_sat) * 80, 5, 35)) * prof["vib"]
    saturation = float(np.clip((0.30 - s.mean_sat) * 30, -10, 10) + prof["sat"] * 30)
    if s.mean_sat < 0.25:
        notes.append("สีซีด → เพิ่มความสดแบบเป็นธรรมชาติ")

    # Detail
    denoise = float(np.clip((s.noise_sigma - 2.0) * 10, 0, 35)) * prof["den"]
    if denoise > 5:
        notes.append("มีนอยส์ → ลดนอยส์แบบรักษาผิว")
    else:
        denoise = 0.0
    sharp_base = 45 if s.sharpness_score < 150 else 30 if s.sharpness_score < 600 else 18
    sharpness = float(sharp_base * prof["sharp"])
    if denoise > 0:
        sharpness *= 0.8
    notes.append("เพิ่มความคมชัดแบบไม่ให้เกิดขอบขาว")

    s.notes = notes
    adj = Adjustments(
        exposure=round(ev, 2), brightness=0.0, contrast=round(contrast),
        highlights=round(highlights), shadows=round(shadows),
        temperature=round(temperature), tint=round(tint),
        vibrance=round(vibrance), saturation=round(saturation),
        sharpness=round(sharpness), denoise=round(denoise),
    ).clamped()
    return adj, s


def make_preview(img: Image.Image, max_side: int = 1400) -> Image.Image:
    """Downscaled copy for fast on-screen preview (never used for export)."""
    prev = img.copy()
    prev.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return prev


def before_after_split(before: Image.Image, after: Image.Image, position: float = 0.5) -> Image.Image:
    """Compose a single image: left part = before, right part = after, with a divider."""
    a = before.convert("RGB")
    b = after.convert("RGB").resize(a.size, Image.Resampling.LANCZOS)
    wa = np.asarray(a).copy()
    wb = np.asarray(b)
    x = int(np.clip(position, 0, 1) * wa.shape[1])
    wa[:, x:] = wb[:, x:]
    line = max(2, wa.shape[1] // 400)
    wa[:, max(0, x - line // 2): x + line // 2 + 1] = 255
    return Image.fromarray(wa)
