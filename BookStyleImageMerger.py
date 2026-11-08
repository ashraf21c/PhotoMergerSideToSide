import io
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError
from streamlit_cropper import st_cropper
from localization import LANGUAGES, translate


APP_DIR = Path(__file__).resolve().parent
APP_ICON_PATH = APP_DIR / "app_icon.png"


def detect_language(locale: str | None) -> str:
    if locale:
        language_code = locale.split("-", maxsplit=1)[0].lower()
        if language_code in LANGUAGES:
            return language_code
    return "en"


if "language" not in st.session_state:
    st.session_state.language = detect_language(st.context.locale)

language = st.session_state.language
st.set_page_config(
    layout="wide",
    page_title=translate(language, "Book-Style Image Merger"),
    page_icon=str(APP_ICON_PATH),
)


def tr(text: str) -> str:
    return translate(language, text)


st.selectbox(
    tr("Language"),
    options=list(LANGUAGES),
    format_func=lambda code: LANGUAGES[code],
    key="language",
)
language = st.session_state.language
if language == "ar":
    st.markdown(
        "<style>"
        "[data-testid='stAppViewContainer'] { direction: rtl; }"
        "[data-testid='stSidebar'] { direction: rtl; }"
        "</style>",
        unsafe_allow_html=True,
    )


def detect_page_edges(
    image: Image.Image,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = grayscale.shape
    if width < 2:
        edge = max(0, width - 1)
        return (0, 0, 0), (edge, edge, edge)

    left_edges = []
    right_edges = []

    for fraction in (0.1, 0.5, 0.9):
        y = int(height * fraction)
        row = grayscale[max(0, y - 8) : min(height, y + 9)].mean(axis=0)
        center = np.percentile(row[int(width * 0.2) : int(width * 0.8)], 70)
        threshold = max(45, center * 0.65)
        left_edge = next(
            (
                x
                for x in range(width // 2)
                if row[x : min(x + 5, width // 2)].mean() > threshold
            ),
            0,
        )
        right_edge = next(
            (
                x
                for x in range(width - 1, width // 2, -1)
                if row[max(x - 4, width // 2) : x + 1].mean() > threshold
            ),
            width - 1,
        )
        left_edges.append(left_edge)
        right_edges.append(right_edge)

    return (
        (left_edges[0], left_edges[1], left_edges[2]),
        (right_edges[0], right_edges[1], right_edges[2]),
    )


def _edge_position(
    y: float,
    height: int,
    positions: tuple[int, int, int],
) -> float:
    top, middle, bottom = positions
    midpoint = (height - 1) / 2
    if y <= midpoint:
        fraction = y / midpoint if midpoint else 0
        return top + (middle - top) * fraction
    fraction = (y - midpoint) / (height - 1 - midpoint)
    return middle + (bottom - middle) * fraction


def flatten_curved_edge(
    image: Image.Image,
    side: Literal["Left", "Right"],
    positions: tuple[int, int, int],
    strength: int = 100,
) -> Image.Image:
    width, height = image.size
    if width < 2 or height < 2:
        return image.copy()
    if not 0 <= strength <= 100:
        raise ValueError("Flattening strength must be between 0 and 100.")
    if side == "Left" and any(position >= width - 1 for position in positions):
        raise ValueError("Left-edge positions must leave space for the page.")
    if side == "Right" and any(position <= 0 for position in positions):
        raise ValueError("Right-edge positions must leave space for the page.")

    band_count = min(128, height)
    output_y = [
        round(index * height / band_count)
        for index in range(band_count + 1)
    ]
    source_y = [
        index * (height - 1) / band_count
        for index in range(band_count + 1)
    ]

    def page_bounds(y: float) -> tuple[float, float]:
        edge = _edge_position(y, height, positions)
        if side == "Left":
            return edge, width - 1
        return 0, edge

    bounds = [page_bounds(y) for y in source_y]
    row_widths = [right - left for left, right in bounds]
    output_width = max(1, round(max(row_widths)) + 1)
    if any(row_width <= 0 for row_width in row_widths):
        raise ValueError("The edge positions must leave a positive page width.")

    mesh = []
    for output_y0, output_y1, source_y0, source_y1, (left0, right0), (
        left1,
        right1,
    ) in zip(
        output_y,
        output_y[1:],
        source_y,
        source_y[1:],
        bounds,
        bounds[1:],
    ):
        mesh.append(
            (
                (0, output_y0, output_width, output_y1),
                (
                    left0,
                    source_y0,
                    left1,
                    source_y1,
                    right1,
                    source_y1,
                    right0,
                    source_y0,
                ),
            )
        )

    flattened = image.transform(
        (output_width, height),
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
    )
    if strength == 100:
        return flattened

    flattened = flattened.resize(image.size, Image.Resampling.LANCZOS)
    return Image.blend(
        image.convert("RGB"),
        flattened.convert("RGB"),
        strength / 100,
    )


def make_gallery(
    images: list[Image.Image],
    columns: int,
    horizontal_gap: int,
    vertical_gap: int,
    background: tuple[int, int, int],
    alignment: str,
    direction: Literal["Left to right", "Right to left"] = "Left to right",
) -> Image.Image:
    rows = [
        images[start : start + columns]
        for start in range(0, len(images), columns)
    ]
    row_sizes = [
        (
            sum(image.width for image in row)
            + horizontal_gap * (len(row) - 1),
            max(image.height for image in row),
        )
        for row in rows
    ]
    canvas_width = max(row_width for row_width, _ in row_sizes)
    canvas_height = sum(row_height for _, row_height in row_sizes)
    canvas_height += vertical_gap * (len(rows) - 1)
    gallery = Image.new("RGB", (canvas_width, canvas_height), background)

    y = 0
    for row, (_, row_height) in zip(rows, row_sizes):
        row_width = sum(image.width for image in row)
        row_width += horizontal_gap * (len(row) - 1)
        x = (canvas_width - row_width) // 2
        row_images = row if direction == "Left to right" else reversed(row)
        for image in row_images:
            if alignment == "center":
                y_offset = (row_height - image.height) // 2
            elif alignment == "bottom":
                y_offset = row_height - image.height
            else:
                y_offset = 0
            mask = image.getchannel("A") if "A" in image.getbands() else None
            gallery.paste(image, (x, y + y_offset), mask)
            x += image.width + horizontal_gap
        y += row_height + vertical_gap

    return gallery


def fit_images_to_uniform_size(
    images: list[Image.Image],
    background: tuple[int, int, int],
) -> tuple[list[Image.Image], tuple[int, int], list[int]]:
    page_bounds = []
    failed_indices = []
    for index, image in enumerate(images):
        bounds = detect_page_bounds(image)
        if bounds is None:
            failed_indices.append(index)
            bounds = (0, 0, image.width, image.height)
        page_bounds.append(bounds)

    reference_index = min(
        range(len(images)),
        key=lambda index: (
            (page_bounds[index][2] - page_bounds[index][0])
            * (page_bounds[index][3] - page_bounds[index][1])
        ),
    )
    reference_bounds = page_bounds[reference_index]
    target_width = reference_bounds[2] - reference_bounds[0]
    target_height = reference_bounds[3] - reference_bounds[1]

    scaled_images = []
    scaled_page_centers = []
    for image, (left, top, right, bottom) in zip(images, page_bounds):
        page_width = right - left
        page_height = bottom - top
        scale = min(target_width / page_width, target_height / page_height, 1.0)
        scaled_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        scaled_image = image.convert("RGB").resize(
            scaled_size,
            Image.Resampling.LANCZOS,
        )
        scaled_images.append(scaled_image)
        scaled_page_centers.append(
            (
                (left + right) * scale / 2,
                (top + bottom) * scale / 2,
            )
        )

    left_extent = max(center[0] for center in scaled_page_centers)
    right_extent = max(
        image.width - center[0]
        for image, center in zip(scaled_images, scaled_page_centers)
    )
    top_extent = max(center[1] for center in scaled_page_centers)
    bottom_extent = max(
        image.height - center[1]
        for image, center in zip(scaled_images, scaled_page_centers)
    )
    common_size = (
        max(1, math.ceil(left_extent + right_extent)),
        max(1, math.ceil(top_extent + bottom_extent)),
    )

    fitted_images = []
    for image, (center_x, center_y) in zip(
        scaled_images,
        scaled_page_centers,
    ):
        tile = Image.new("RGB", common_size, background)
        position = (
            round(left_extent - center_x),
            round(top_extent - center_y),
        )
        tile.paste(image, position)
        fitted_images.append(tile)
    return fitted_images, common_size, failed_indices


def detect_page_bounds(
    image: Image.Image,
) -> tuple[int, int, int, int] | None:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    detection_scale = min(1.0, 1000 / max(width, height))
    if detection_scale < 1:
        small = cv2.resize(
            rgb,
            None,
            fx=detection_scale,
            fy=detection_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb

    grayscale = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    small_height, small_width = grayscale.shape
    image_area = small_width * small_height
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08 or area > image_area * 0.92:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        aspect_ratio = box_width / box_height
        if not 0.4 <= aspect_ratio <= 2.5:
            continue
        rectangularity = area / (box_width * box_height)
        candidates.append((area * rectangularity, x, y, box_width, box_height))

    if not candidates:
        return None

    _, x, y, box_width, box_height = max(candidates)
    inverse_scale = 1 / detection_scale
    left = max(0, round(x * inverse_scale))
    top = max(0, round(y * inverse_scale))
    right = min(width, round((x + box_width) * inverse_scale))
    bottom = min(height, round((y + box_height) * inverse_scale))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def encode_gallery(
    image: Image.Image,
    format_name: Literal["PNG", "JPEG"],
    quality: int = 95,
    max_edge: int | None = None,
) -> io.BytesIO:
    output = io.BytesIO()
    export_image = image
    if max_edge is not None and max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        export_image = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    if format_name == "JPEG":
        export_image = export_image.convert("RGB")
        export_image.save(
            output,
            format=format_name,
            quality=quality,
            optimize=True,
            progressive=True,
        )
    else:
        export_image.save(output, format=format_name)
    output.seek(0)
    return output


st.title(tr("Book-Style Image Merger & Editor"))
st.caption(
    tr(
        "Upload one or more pages. Crop and straighten each image independently, "
        "then arrange them as a spread or multi-row gallery."
    )
)

uploads = st.file_uploader(
    tr("Upload pages"),
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="image_uploads",
)

if not uploads:
    st.info(tr("Upload one or more images to get started."))
    st.stop()

ordered_images = []
for index, upload in enumerate(uploads):
    try:
        upload.seek(0)
        with Image.open(upload) as source:
            image = ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, OSError) as error:
        st.error(
            tr("Could not open {filename}: {error}").format(
                filename=upload.name,
                error=error,
            )
        )
        st.stop()

    with st.expander(
        tr("{number}. {filename}").format(number=index + 1, filename=upload.name),
        expanded=True,
    ):
        st.image(
            image,
            caption=tr("Original: {width} × {height}").format(
                width=image.width,
                height=image.height,
            ),
        )

        image_order = int(
            st.number_input(
                tr("Image order"),
                min_value=1,
                max_value=len(uploads),
                value=index + 1,
                step=1,
                key=f"image_order_{index}",
                help=tr(
                    "Set a unique number to control this image's gallery order."
                ),
            )
        )

        if st.checkbox(tr("Crop this image"), key=f"crop_enabled_{index}"):
            image = st_cropper(
                image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=None,
                key=f"cropper_{index}",
            )

        if st.checkbox(
            tr("Flatten a curved page edge"),
            key=f"flatten_enabled_{index}",
            help=(
                tr(
                    "Adjust the selected page edge at the top, middle, and bottom. "
                    "The image is remapped row by row to straighten that edge."
                )
            ),
        ):
            detected_left, detected_right = detect_page_edges(image)
            side = st.selectbox(
                tr("Curved edge"),
                ["Left", "Right"],
                format_func=tr,
                key=f"flatten_side_{index}",
            )
            width, height = image.size
            edge_defaults = detected_left if side == "Left" else detected_right
            points = st.columns(3)
            positions = []
            labels = ("Top", "Middle", "Bottom")
            for point, label, edge_default in zip(
                points,
                labels,
                edge_defaults,
            ):
                with point:
                    positions.append(
                        int(
                            st.number_input(
                                tr(f"{label} edge x"),
                                min_value=0,
                                max_value=width - 1,
                                value=int(edge_default),
                                key=(
                                    f"flatten_{label.lower()}_"
                                    f"{side.lower()}_{index}"
                                ),
                            )
                        )
                    )
            flatten_strength = int(
                st.slider(
                    tr("Flattening strength"),
                    min_value=0,
                    max_value=100,
                    value=100,
                    step=5,
                    format="%d%%",
                    key=f"flatten_strength_{index}",
                    help=(
                        tr(
                            "0% keeps the original; 100% applies the full "
                            "edge-straightening correction. Intermediate values "
                            "blend between them."
                        )
                    ),
                )
            )
            guide = image.convert("RGB").copy()
            guide_draw = ImageDraw.Draw(guide)
            guide_points = [
                (x, y)
                for x, y in zip(
                    positions,
                    (0, (height - 1) // 2, height - 1),
                )
            ]
            guide_draw.line(guide_points, fill="#FF0000", width=4)
            for x, y in guide_points:
                radius = max(4, min(width, height) // 100)
                guide_draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill="#FF0000",
                )
            st.image(
                guide,
                caption=(
                    tr(
                        "Red line: estimated {side} page edge. The x controls "
                        "use pixels from the image's left side."
                    ).format(side=tr(side).lower())
                ),
            )
            try:
                image = flatten_curved_edge(
                    image,
                    side,
                    tuple(positions),
                    flatten_strength,
                )
            except ValueError as error:
                st.error(
                    tr("Could not flatten {filename}: {error}").format(
                        filename=upload.name,
                        error=error,
                    )
                )
                st.stop()

        if st.checkbox(
            tr("Apply post-processing to this image"),
            key=f"edit_enabled_{index}",
        ):
            st.markdown(f"**{tr('Image post-processing')}**")
            scale = st.slider(
                tr("Scale factor"),
                1.0,
                4.0,
                1.0,
                0.1,
                key=f"image_scale_{index}",
            )
            if scale != 1.0:
                image = image.resize(
                    (
                        max(1, int(image.width * scale)),
                        max(1, int(image.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            if st.checkbox(
                tr("Set exact dimensions for this image"),
                key=f"image_exact_dimensions_{index}",
            ):
                dimension_col, height_col = st.columns(2)
                with dimension_col:
                    target_width = int(
                        st.number_input(
                            tr("Image width"),
                            min_value=1,
                            value=image.width,
                            key=f"image_width_{index}",
                        )
                    )
                with height_col:
                    target_height = int(
                        st.number_input(
                            tr("Image height"),
                            min_value=1,
                            value=image.height,
                            key=f"image_height_{index}",
                        )
                    )
                if (target_width, target_height) != image.size:
                    image = image.resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    )

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                brightness = st.slider(
                    tr("Brightness"),
                    0.5,
                    2.0,
                    1.0,
                    0.05,
                    key=f"image_brightness_{index}",
                )
                contrast = st.slider(
                    tr("Contrast"),
                    0.5,
                    2.0,
                    1.0,
                    0.05,
                    key=f"image_contrast_{index}",
                )
            with edit_col2:
                sharpness = st.slider(
                    tr("Sharpness"),
                    0.0,
                    2.0,
                    1.0,
                    0.05,
                    key=f"image_sharpness_{index}",
                )
                saturation = st.slider(
                    tr("Saturation"),
                    0.0,
                    2.0,
                    1.0,
                    0.05,
                    key=f"image_saturation_{index}",
                )

            if brightness != 1.0:
                image = ImageEnhance.Brightness(image).enhance(brightness)
            if contrast != 1.0:
                image = ImageEnhance.Contrast(image).enhance(contrast)
            if sharpness != 1.0:
                image = ImageEnhance.Sharpness(image).enhance(sharpness)
            if saturation != 1.0:
                image = ImageEnhance.Color(image).enhance(saturation)

        st.caption(
            tr("Working size: {width} × {height}").format(
                width=image.width,
                height=image.height,
            )
        )
        ordered_images.append((image_order, index, image))

order_values = [order for order, _, _ in ordered_images]
if len(set(order_values)) != len(order_values):
    st.error(
        tr(
            "Image order numbers must be unique. Assign each image a different number."
        )
    )
    st.stop()

ordered_images.sort(key=lambda item: item[0])
processed_images = [image for _, _, image in ordered_images]

st.divider()
st.subheader(tr("📐 Layout"))

layout_col, gap_col, align_col, direction_col = st.columns(4)
with layout_col:
    images_per_row = st.selectbox(
        tr("Images per row"),
        options=[1, 2, 3, 4],
        index=3,
        help=(
            tr(
                "Keeps images at their original size. More images continue on "
                "subsequent gallery rows."
            )
        ),
        key="images_per_row",
    )
with gap_col:
    horizontal_gap = int(
        st.number_input(
            tr("Horizontal spacing (px)"),
            min_value=0,
            max_value=500,
            value=20,
            step=1,
            key="horizontal_spacing",
        )
    )
with align_col:
    alignment = st.selectbox(
        tr("Vertical alignment"),
        ["top", "center", "bottom"],
        index=0,
        format_func=tr,
        key="vertical_alignment",
    )
with direction_col:
    direction = st.selectbox(
        tr("Reading direction"),
        ["Left to right", "Right to left"],
        index=0,
        format_func=tr,
        help=tr(
            "Right-to-left places the lowest image order at the right side "
            "of each row."
        ),
        key="reading_direction",
    )

unify_dimensions = st.checkbox(
    tr("Make all images the same size (no cropping)"),
    key="unify_image_dimensions",
    help=tr(
        "Detects page boundaries, scales pages to a common size, and keeps "
        "the full photos and their backgrounds. Images are not cropped or enlarged."
    ),
)

vertical_gap = int(
    st.number_input(
        tr("Space between rows (px)"),
        min_value=0,
        max_value=500,
        value=20,
        step=1,
        key="vertical_spacing",
    )
)
background_choice = st.color_picker(
    tr("Gallery background color"),
    "#FFFFFF",
    key="gallery_background",
)
background = tuple(
    int(background_choice[index : index + 2], 16)
    for index in (1, 3, 5)
)

if unify_dimensions:
    processed_images, common_size, undetected_indices = fit_images_to_uniform_size(
        processed_images,
        background,
    )
    st.caption(
        tr(
            "Detected page areas are normalized; full photos and backgrounds "
            "are retained in {width} × {height} px frames."
        ).format(width=common_size[0], height=common_size[1])
    )
    if undetected_indices:
        undetected_names = ", ".join(
            uploads[ordered_images[index][1]].name
            for index in undetected_indices
        )
        st.warning(
            tr(
                "Could not detect page boundaries in: {filenames}. "
                "Those images are normalized using their full image bounds."
            ).format(filenames=undetected_names)
        )

gallery = make_gallery(
    processed_images,
    columns=images_per_row,
    horizontal_gap=horizontal_gap,
    vertical_gap=vertical_gap,
    background=background,
    alignment=alignment,
    direction=direction,
)
st.success(
    tr(
        "Gallery: {count} image(s), up to {columns} per row — "
        "{width} × {height} px"
    ).format(
        count=len(processed_images),
        columns=images_per_row,
        width=gallery.width,
        height=gallery.height,
    )
)

st.divider()
st.subheader(tr("🛠️ Post-Processing"))

working_image = gallery
scale = st.slider(
    tr("Scale factor"),
    1.0,
    4.0,
    1.0,
    0.1,
    key="gallery_scale",
)
if scale != 1.0:
    working_image = working_image.resize(
        (
            int(working_image.width * scale),
            int(working_image.height * scale),
        ),
        Image.Resampling.LANCZOS,
    )

dimension_col, height_col = st.columns(2)
with dimension_col:
    target_width = int(
        st.number_input(
            tr("Final width"),
            min_value=1,
            value=working_image.width,
            key="final_width",
        )
    )
with height_col:
    target_height = int(
        st.number_input(
            tr("Final height"),
            min_value=1,
            value=working_image.height,
            key="final_height",
        )
    )
if (target_width, target_height) != working_image.size:
    working_image = working_image.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

edit_col1, edit_col2 = st.columns(2)
with edit_col1:
    brightness = st.slider(
        tr("Brightness"),
        0.5,
        2.0,
        1.0,
        0.05,
        key="gallery_brightness",
    )
    contrast = st.slider(
        tr("Contrast"),
        0.5,
        2.0,
        1.0,
        0.05,
        key="gallery_contrast",
    )
with edit_col2:
    sharpness = st.slider(
        tr("Sharpness"),
        0.0,
        2.0,
        1.0,
        0.05,
        key="gallery_sharpness",
    )
    saturation = st.slider(
        tr("Saturation"),
        0.0,
        2.0,
        1.0,
        0.05,
        key="gallery_saturation",
    )

if brightness != 1.0:
    working_image = ImageEnhance.Brightness(working_image).enhance(brightness)
if contrast != 1.0:
    working_image = ImageEnhance.Contrast(working_image).enhance(contrast)
if sharpness != 1.0:
    working_image = ImageEnhance.Sharpness(working_image).enhance(sharpness)
if saturation != 1.0:
    working_image = ImageEnhance.Color(working_image).enhance(saturation)

st.divider()
st.subheader(tr("📸 Preview & Download"))
st.image(working_image, width="stretch", alt=tr("Final gallery preview"))

png_download = encode_gallery(working_image, "PNG").getvalue()
jpg_download = encode_gallery(working_image, "JPEG", quality=95).getvalue()
web_download = encode_gallery(
    working_image,
    "JPEG",
    quality=82,
    max_edge=2560,
).getvalue()

download_col1, download_col2, download_col3 = st.columns(3)
with download_col1:
    png_download_clicked = st.download_button(
        tr("⬇️ Download Final Gallery (PNG)"),
        data=png_download,
        file_name="book_gallery.png",
        mime="image/png",
        help=tr("Downloads the full-size gallery as a lossless PNG."),
        width="stretch",
    )
with download_col2:
    jpg_download_clicked = st.download_button(
        tr("⬇️ Download Final Gallery (JPG)"),
        data=jpg_download,
        file_name="book_gallery.jpg",
        mime="image/jpeg",
        help=tr("Saves the full-size gallery as a high-quality JPEG."),
        width="stretch",
    )
with download_col3:
    web_download_clicked = st.download_button(
        tr("⬇️ Compress for Web and Download"),
        data=web_download,
        file_name="book_gallery_web.jpg",
        mime="image/jpeg",
        help=tr(
            "Resizes the longest edge to at most 2560 px and saves as JPEG "
            "at 82% quality."
        ),
        width="stretch",
    )

if png_download_clicked or jpg_download_clicked or web_download_clicked:
    st.success(tr("Download started. Check your browser's downloads."))
