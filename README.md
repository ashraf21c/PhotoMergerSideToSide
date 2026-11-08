# Book-Style Image Merger

Start the app by double-clicking `run_book_style_merger.bat`. Upload one or
more JPG, PNG, or WebP images. Each uploaded image has its own crop, flatten,
and optional post-processing controls.

## Languages

The app detects the browser's preferred language when it starts. English,
Arabic, and Japanese are available from the **Language** menu; choosing one
manually overrides automatic detection for the current browser session.
Arabic uses a right-to-left page direction.

## Straightening a curved page edge

1. Expand the image's settings and enable **Flatten a curved page edge**.
2. Select **Left** or **Right**, whichever page edge is curved.
3. Use the preview's red guide to check the estimated edge. The app detects
   an initial position at the top, middle, and bottom of the image.
4. Adjust **Top edge x**, **Middle edge x**, and **Bottom edge x** if the
   guide does not follow the page edge. These are horizontal pixel positions:
   `0` is the leftmost image pixel, and the maximum is the rightmost pixel.
   The controls refer to the current image after optional cropping. Change
   the three points in small increments and check the red guide after each
   adjustment. For example, on a 1200-pixel-wide image, left-edge values
   `30`, `50`, `32` place the estimated page edge 20 pixels farther right in
   the middle than at the top. For a right edge, values near the image width
   might look like `1170`, `1150`, `1168`. These are example coordinates only;
   use the red guide and your image's actual width to choose the values.
5. Adjust **Flattening strength**:
   - **0%** keeps the original image.
   - **100%** applies the full edge-straightening correction.
   - **50%** blends the original and fully corrected images equally; other
     values blend proportionally.

The correction straightens the selected page edge; it does not automatically
remove all perspective or dewarp curved text. If automatic edge detection
does not match the page, manually move the three points to the visible page
boundary.

## Image order and gallery

Set a unique **Image order** number on each image. The smallest number appears
first; the app sorts the images by these numbers before laying them out. Each
number from 1 to the number of uploaded images must be used exactly once.
Choose **Images per row** to control how many images fit in each gallery row.
Extra images continue on subsequent rows. Set **Reading direction** to
**Right to left** for RTL layouts: the lowest image-order number appears at
the right of the first row, and each row continues toward the left. Later
rows continue below, also starting from the right. **Left to right** keeps the
lowest order at the left of each row.

## Optional image adjustments

Enable **Apply post-processing to this image** to adjust that image's scale,
brightness, contrast, sharpness, and saturation before assembling the
gallery. **Set exact dimensions for this image** is optional. The
post-processing section below the gallery can also adjust the assembled
result.

Enable **Make all images the same size (no cropping)** in the layout controls
to detect each page boundary and scale the visible pages to a common size.
The entire photo, including its original background, is retained; photos are
not cropped or enlarged. The shared canvas is sized to contain every
normalized photo, and any extra canvas area uses the selected gallery
background. If a page boundary cannot be detected, the app displays a warning
and uses that image's full bounds for sizing.

## Download formats

- **PNG** preserves the full-size gallery without JPEG compression.
- **JPG** exports the full-size gallery as a high-quality JPEG.
- **Compress for Web and Download** creates a JPEG at 82% quality and scales
  the longest edge down to at most 2560 pixels.

## App icon

The app uses `app_icon.png` as its Streamlit page icon (shown as the browser
favicon and alongside the page title in the browser). `app_icon.ico` is the
matching Windows icon for a shortcut to the app. A Streamlit app runs inside
the browser, so the browser controls its own native window and taskbar icon;
the page cannot replace those icons in a normal browser window.

## Adding another UI language

1. Open `localization.py`.
2. Add the language code and its native name to `LANGUAGES`, for example
   `"fr": "Français"`.
3. Add a `"fr"` entry to `TRANSLATIONS`. Translate each English source string
   used by the app into the corresponding value. Keep `{placeholders}` such
   as `{width}` and `{filename}` unchanged so formatted messages continue to
   work.
4. Add the language code to `detect_language` in
   `BookStyleImageMerger.py` if its browser locale should be detected
   automatically. Languages not added there can still be selected manually.
5. Add a translated README if desired, then restart the app and check every
   control, error message, and formatted caption in the new language.

Untranslated or missing entries fall back to their English source text.
