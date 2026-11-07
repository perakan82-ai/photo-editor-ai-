import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _gradient(w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    base = np.concatenate([x * 0.8 + 0.1 + 0 * y, y * 0.6 + 0.2 + 0 * x, (1 - x) * 0.5 + 0.2 + 0 * y], axis=2)
    base = base + rng.normal(0, 0.03, base.shape).astype(np.float32)
    return Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8), "RGB")


@pytest.fixture
def rgb_image():
    return _gradient()


@pytest.fixture
def dark_image():
    arr = np.asarray(_gradient()).astype(np.float32) * 0.25
    return Image.fromarray(arr.astype(np.uint8), "RGB")


@pytest.fixture
def make_bytes():
    def _make(img, fmt="JPEG", **kw):
        buf = io.BytesIO()
        img.save(buf, format=fmt, **kw)
        return buf.getvalue()
    return _make
