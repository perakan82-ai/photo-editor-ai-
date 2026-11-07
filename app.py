"""
app.py - Professional Photo Editor AI (Streamlit UI)

Run:  streamlit run app.py
All image processing lives in image_engine.py; this file is UI only.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from export_utils import (EXPORT_FORMATS, JPEG_QUALITY, MIME, default_export_format,
                          encode_image, output_filename, realesrgan_available)
from image_engine import (MODES, PARAM_RANGES, Adjustments, apply_adjustments,
                          auto_params, before_after_split, make_preview)
from validators import MAX_FILE_BYTES, ValidationError, validate_and_load

st.set_page_config(page_title="Professional Photo Editor AI", page_icon="📸",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Styling (mobile friendly)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1200px;}
      .app-title {font-size: 1.9rem; font-weight: 750; margin-bottom: 0;}
      .app-sub {color: #8a8f98; margin-top: .2rem;}
      div.stButton > button[kind="primary"] {
          width: 100%; height: 3.2rem; font-size: 1.1rem; font-weight: 700;
          border-radius: 12px;}
      div.stDownloadButton > button {
          width: 100%; height: 3rem; font-weight: 700; border-radius: 12px;}
      .note-chip {display:inline-block; padding:.25rem .6rem; margin:.15rem;
          border-radius: 999px; background: rgba(120,120,255,.12); font-size:.85rem;}
      @media (max-width: 640px) { .app-title {font-size: 1.45rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)

SLIDER_LABELS = {
    "exposure": "Exposure (EV)", "brightness": "Brightness", "contrast": "Contrast",
    "highlights": "Highlights", "shadows": "Shadows", "temperature": "Temperature",
    "tint": "Tint", "vibrance": "Vibrance", "saturation": "Saturation",
    "sharpness": "Sharpness", "denoise": "Denoise",
}


def show_image(target, img, caption=None):
    """st.image full width; works on both old and new Streamlit APIs."""
    try:
        target.image(img, caption=caption, width="stretch")
    except Exception:
        target.image(img, caption=caption, use_container_width=True)


def _key(name: str) -> str:
    return f"p_{name}"


def reset_sliders():
    for name, (_, _, default) in PARAM_RANGES.items():
        st.session_state[_key(name)] = default


def current_adjustments() -> Adjustments:
    return Adjustments(**{n: float(st.session_state.get(_key(n), PARAM_RANGES[n][2]))
                          for n in PARAM_RANGES})


def on_generate():
    """Analyse the photo and fill sliders with the chosen mode's settings."""
    loaded = st.session_state.get("loaded")
    if loaded is None:
        return
    mode = st.session_state.get("mode", MODES[0])
    if mode == "Manual Adjustment":
        st.session_state["notes"] = ["โหมด Manual: ใช้ค่าจากแถบปรับด้านล่าง"]
    else:
        adj, stats = auto_params(loaded.image, mode)
        for name, value in adj.to_dict().items():
            lo, hi, _ = PARAM_RANGES[name]
            st.session_state[_key(name)] = float(value) if isinstance(lo, float) else int(round(value))
        st.session_state["notes"] = stats.notes
    st.session_state["generated"] = True


# ----------------------------------------------------------------------------
# Header + upload
# ----------------------------------------------------------------------------
st.markdown('<p class="app-title">📸 Professional Photo Editor AI</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">อัปโหลดรูป → กด Generate → ระบบวิเคราะห์และปรับแสง สี ความคมชัดให้อัตโนมัติ → ดาวน์โหลด</p>',
            unsafe_allow_html=True)

uploaded = st.file_uploader("อัปโหลดรูปภาพ (JPG, JPEG, PNG, WEBP)",
                            type=["jpg", "jpeg", "png", "webp"],
                            help=f"ขนาดไม่เกิน {MAX_FILE_BYTES // (1024*1024)} MB")

if uploaded is None:
    st.info("เริ่มต้นด้วยการอัปโหลดรูปภาพจากมือถือหรือคอมพิวเตอร์")
    st.stop()

data = uploaded.getvalue()
file_id = hashlib.sha256(data).hexdigest()

if st.session_state.get("file_id") != file_id:
    try:
        loaded = validate_and_load(data, uploaded.name)
    except ValidationError as e:
        st.error(f"❌ {e}")
        st.stop()
    st.session_state.update(file_id=file_id, loaded=loaded, generated=False, notes=[],
                            result_cache={}, preview_src=make_preview(loaded.image))
    reset_sliders()

loaded = st.session_state["loaded"]
st.caption(f"{uploaded.name} · {loaded.format} · {loaded.width}×{loaded.height}px · "
           f"{loaded.file_size / 1024 / 1024:.2f} MB · โหมดสี {loaded.original_mode}")

# ----------------------------------------------------------------------------
# Mode + Generate
# ----------------------------------------------------------------------------
col_m, col_g = st.columns([2, 1])
with col_m:
    st.selectbox("โหมดการแต่งภาพ", MODES, key="mode",
                 help="Auto Professional: สมดุลทั่วไป · Natural Portrait: ถนอมผิว สีอ่อนโยน · "
                      "Event & Social: สีสดและคมขึ้น · Manual: ปรับเองทั้งหมด")
with col_g:
    st.button("✨ Generate", type="primary", on_click=on_generate)

if st.session_state.get("notes"):
    st.markdown("".join(f'<span class="note-chip">{n}</span>' for n in st.session_state["notes"]),
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sliders (auto-filled after Generate, editable any time)
# ----------------------------------------------------------------------------
manual = st.session_state.get("mode") == "Manual Adjustment"
with st.expander("🎚️ ปรับละเอียด (Fine-tune)", expanded=manual):
    groups = [("แสง", ["exposure", "brightness", "contrast", "highlights", "shadows"]),
              ("สี", ["temperature", "tint", "vibrance", "saturation"]),
              ("รายละเอียด", ["sharpness", "denoise"])]
    cols = st.columns(3)
    for col, (title, names) in zip(cols, groups):
        with col:
            st.markdown(f"**{title}**")
            for n in names:
                lo, hi, _ = PARAM_RANGES[n]
                step = 0.05 if isinstance(lo, float) else 1
                st.slider(SLIDER_LABELS[n], lo, hi, key=_key(n), step=step)
    st.button("↺ รีเซ็ตทั้งหมด", on_click=reset_sliders)

# ----------------------------------------------------------------------------
# Processing (cached per parameter set)
# ----------------------------------------------------------------------------
adj = current_adjustments().clamped()
params_key = tuple(round(v, 4) for v in adj.to_dict().values())
cache = st.session_state.setdefault("result_cache", {})

if params_key not in cache:
    with st.spinner("กำลังปรับภาพความละเอียดเต็ม..."):
        try:
            full = apply_adjustments(loaded.image, adj)
        except MemoryError:
            st.error("หน่วยความจำไม่พอสำหรับรูปขนาดนี้ ลองใช้รูปที่เล็กลง")
            st.stop()
        except Exception as e:  # never crash the app
            st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")
            st.stop()
    cache.clear()  # keep only the latest result to limit memory
    cache[params_key] = {"full": full, "preview": make_preview(full)}

result = cache[params_key]
before_prev = st.session_state["preview_src"]
after_prev = result["preview"]

# ----------------------------------------------------------------------------
# Preview
# ----------------------------------------------------------------------------
if adj.is_identity():
    st.warning("ยังไม่มีการปรับแต่ง — กด ✨ Generate หรือขยับแถบปรับ")

tab1, tab2 = st.tabs(["🖼️ ก่อน / หลัง", "↔️ สไลด์เปรียบเทียบ"])
with tab1:
    c1, c2 = st.columns(2)
    show_image(c1, before_prev, "ต้นฉบับ (Original)")
    show_image(c2, after_prev, "หลังแต่ง (Edited)")
with tab2:
    pos = st.slider("ตำแหน่งเส้นแบ่ง (ซ้าย = ต้นฉบับ, ขวา = หลังแต่ง)", 0, 100, 50, key="split")
    show_image(st, before_after_split(before_prev, after_prev, pos / 100))

# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
st.subheader("💾 Export")
has_alpha = result["full"].mode == "RGBA"
default_fmt = default_export_format(loaded.format, has_alpha)

e1, e2, e3 = st.columns(3)
with e1:
    out_mode = st.radio("ความละเอียด", ["Original Resolution", "AI Upscale 2x"], index=0)
with e2:
    fmt = st.selectbox("ไฟล์", EXPORT_FORMATS, index=EXPORT_FORMATS.index(default_fmt))
with e3:
    quality = st.slider("คุณภาพ JPEG/WEBP", 85, 100, JPEG_QUALITY, disabled=(fmt == "PNG"))

if out_mode == "AI Upscale 2x":
    if realesrgan_available():
        st.info("ตรวจพบแพ็กเกจ Real-ESRGAN แต่ฟีเจอร์ AI Upscale ยังอยู่ในแผนเฟสถัดไป ยังไม่เปิดใช้งาน")
    else:
        st.warning("AI Upscale ยังไม่พร้อมใช้งาน: ไม่ได้ติดตั้งโมเดล Real-ESRGAN "
                   "(ระบบจะไม่ใช้การขยายภาพธรรมดามาแอบอ้างเป็น AI) — ดาวน์โหลดด้านล่างเป็นความละเอียดเดิม")

if fmt == "JPEG" and has_alpha:
    st.caption("JPEG ไม่รองรับพื้นโปร่งใส — พื้นหลังจะถูกเติมเป็นสีขาว")

dl_key = (params_key, fmt, quality)
if st.session_state.get("dl_key") != dl_key:
    st.session_state["dl_bytes"] = encode_image(result["full"], fmt, quality,
                                                exif=loaded.exif, icc_profile=loaded.icc_profile)
    st.session_state["dl_key"] = dl_key

out_bytes = st.session_state["dl_bytes"]
st.download_button(
    f"⬇️ ดาวน์โหลดรูปที่แต่งแล้ว ({result['full'].width}×{result['full'].height}px, "
    f"{len(out_bytes) / 1024 / 1024:.2f} MB)",
    data=out_bytes, file_name=output_filename(uploaded.name, fmt), mime=MIME[fmt],
    type="primary",
)
st.caption("ไฟล์ต้นฉบับของคุณไม่ถูกแก้ไข — ระบบประมวลผลบนสำเนาแยกเสมอ และไม่ส่งรูปไปเซิร์ฟเวอร์ภายนอก")
