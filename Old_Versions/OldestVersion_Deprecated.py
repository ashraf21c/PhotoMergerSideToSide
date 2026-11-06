# pip install streamlit streamlit-cropper Pillow
import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper
import io

st.set_page_config(layout="wide", page_title="Image Merger & Editor")
st.title("🖼️ Image Merger & Editor (with Cropping)")

# ---------- 1. Upload and Merge ----------
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    file1 = st.file_uploader("Upload Image A", type=["jpg", "png", "jpeg", "webp"])
with col_upload2:
    file2 = st.file_uploader("Upload Image B", type=["jpg", "png", "jpeg", "webp"])

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")

    swap = st.checkbox("🔄 Swap left/right (put B on left)", value=False)

    if swap:
        left_img, right_img = img2, img1
    else:
        left_img, right_img = img1, img2

    # Execute merge
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    merged = Image.new("RGB", (w1 + w2, max(h1, h2)), (255, 255, 255))
    merged.paste(left_img, (0, 0))
    merged.paste(right_img, (w1, 0))

    st.success(f"Merged! Original size: {merged.width} x {merged.height}")

    # ---------- 2. Cropping (new) ----------
    st.divider()
    st.subheader("✂️ Crop")
    st.caption("Drag the box to select the crop area. The preview below will update in real time.")

    # Cropping component: no aspect ratio restriction by default
    cropped = st_cropper(
        merged,
        realtime_update=True,
        box_color="#FF0000",
        aspect_ratio=None  # Set to (1,1) etc. for a fixed ratio
    )

    # Use the cropped result as the new working image
    working_img = cropped
    st.write(f"After crop: {working_img.width} x {working_img.height}")

    # ---------- 3. Post-processing Controls ----------
    st.divider()
    st.subheader("🛠️ Post-Processing")

    # Upscaling
    st.markdown("**Upscaling**")
    scale = st.slider("Scale Factor", min_value=1.0, max_value=4.0, value=1.0, step=0.1)
    if scale != 1.0:
        new_w = int(working_img.width * scale)
        new_h = int(working_img.height * scale)
        working_img = working_img.resize((new_w, new_h), Image.LANCZOS)

    # Exact Dimensions
    st.markdown("**Exact Dimensions**")
    col_w, col_h = st.columns(2)
    with col_w:
        target_w = st.number_input("Width", min_value=1, value=working_img.width)
    with col_h:
        target_h = st.number_input("Height", min_value=1, value=working_img.height)

    if target_w != working_img.width or target_h != working_img.height:
        working_img = working_img.resize((int(target_w), int(target_h)), Image.LANCZOS)

    # Basic Adjustments
    st.markdown("**Basic Adjustments**")
    col_edit1, col_edit2 = st.columns(2)
    with col_edit1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
    with col_edit2:
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.05)
        color = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)

    if brightness != 1.0:
        working_img = ImageEnhance.Brightness(working_img).enhance(brightness)
    if contrast != 1.0:
        working_img = ImageEnhance.Contrast(working_img).enhance(contrast)
    if sharpness != 1.0:
        working_img = ImageEnhance.Sharpness(working_img).enhance(sharpness)
    if color != 1.0:
        working_img = ImageEnhance.Color(working_img).enhance(color)

    # ---------- 4. Preview and Download ----------
    st.divider()
    st.subheader("📸 Preview & Download")
    st.image(working_img, use_container_width=True)

    buf = io.BytesIO()
    working_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Final Image (PNG)",
        data=byte_im,
        file_name="merged_cropped_edited.png",
        mime="image/png"
    )
else:
    st.info("👆 Please upload two images to get started.")