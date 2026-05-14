import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
from keras.models import Model
from streamlit_image_comparison import image_comparison
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt

TILE_SIZE = 224  # Each tile is 224x224 pixels

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def add_canvas(image, fill_color=(255, 255, 255)):
    image_width, image_height = image.size
    canvas_width  = image_width  + math.ceil(0.015 * image_width)
    canvas_height = image_height + math.ceil(0.07  * image_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    paste_position = (
        (canvas_width  - image_width)  // 2,
        (canvas_height - image_height) // 7,
    )
    canvas.paste(image, paste_position)
    return canvas

def add_white_border(image, border_size):
    return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))

# ──────────────────────────────────────────────
# Original whole-image prediction (unchanged)
# ──────────────────────────────────────────────

def import_and_predict(image_data, sensitivity=9):
    try:
        original_img = np.array(image_data)
        if original_img.shape[-1] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

        orig_height, orig_width, _ = original_img.shape
        max_dimension     = max(orig_width, orig_height)
        contour_thickness = max(2, int(max_dimension / 200))

        img_resized    = cv2.resize(original_img, (224, 224))
        img_tensor     = np.expand_dims(img_resized, axis=0) / 255.0
        preprocessed_img = img_tensor

        custom_model = Model(
            inputs=model.inputs,
            outputs=(model.layers[sensitivity].output, model.layers[-1].output),
        )
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)

        pred = np.argmax(pred_vec)

        heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height),
                                      interpolation=cv2.INTER_LINEAR)
        heat_map = np.mean(heat_map_resized, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        heat_map = heat_map / heat_map.max()

        threshold     = 0.5
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)
        contours, _   = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        heatmap_image   = Image.fromarray(heatmap_colored)

        contoured_img = original_img.copy()
        cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
        contoured_image = Image.fromarray(contoured_img)

        heatmap_image_rgba  = heatmap_image.convert("RGBA")
        original_img_pil    = Image.fromarray(original_img).convert("RGBA")
        heatmap_overlay     = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

        heatmap_overlay_rgb    = heatmap_overlay.convert("RGB")
        heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
        cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)
        overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

        border_size          = 10
        image_with_border    = add_white_border(image_data, border_size)
        contours_with_border = add_white_border(overlay_img, border_size)

        return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None, None, None


# ──────────────────────────────────────────────
# Tile-based crack detection — sliding window
# ──────────────────────────────────────────────

def build_custom_model(sensitivity=9):
    """Build intermediate-output model for batch inference."""
    return Model(
        inputs=model.inputs,
        outputs=(model.layers[sensitivity].output, model.layers[-1].output),
    )

def predict_tiles_batch(tiles_np, sensitivity=9):
    """
    Run the model on a batch of 224x224 tiles in one forward pass.
    Returns pred_indices (list[int]), pred_vecs (N,3), conv_outputs (N,H,W,C).
    """
    batch = np.stack(tiles_np, axis=0) / 255.0
    custom_model = build_custom_model(sensitivity)
    conv_outputs, pred_vecs = custom_model.predict(batch, verbose=0)
    pred_indices = [int(np.argmax(pv)) for pv in pred_vecs]
    return pred_indices, pred_vecs, conv_outputs


def tiled_crack_detection(image_data, sensitivity=9, progress_bar=None, overlap=0.5):
    """
    Sliding-window tiled crack detection.

    overlap controls how much neighbouring tiles share:
      0.0  = no overlap (fastest, may miss cracks on tile borders)
      0.5  = 50% overlap — tiles shift by TILE_SIZE//2 each step (recommended)
      0.75 = 75% overlap — finest coverage, ~4x more tiles, slower

    For each pixel covered by multiple overlapping tiles the model's
    P(cracked) scores are averaged (score-map fusion), making border
    cracks far more reliably detected.

    The non-overlapping grid shown in the UI uses averaged scores per cell.
    """
    MINI_BATCH_SIZE = 64
    CONF_THRESHOLD  = 95.0   # P(cracked)*100 must exceed this to mark as Cracked

    original_img = np.array(image_data)
    if original_img.shape[-1] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w, _ = original_img.shape
    contour_thickness  = max(2, int(max(orig_w, orig_h) / 200))

    # ── Step size derived from overlap ────────────────────────────────────
    step = max(1, int(TILE_SIZE * (1.0 - overlap)))

    # ── Pad so every sliding-window tile fits fully inside the canvas ─────
    pad_h = (TILE_SIZE - orig_h % step) % step + (TILE_SIZE - step)
    pad_w = (TILE_SIZE - orig_w % step) % step + (TILE_SIZE - step)
    padded_img = cv2.copyMakeBorder(original_img, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_REFLECT)
    pad_h_total, pad_w_total = padded_img.shape[:2]

    # ── Generate all sliding-window tile origins ───────────────────────────
    y_starts = list(range(0, pad_h_total - TILE_SIZE + 1, step))
    x_starts = list(range(0, pad_w_total - TILE_SIZE + 1, step))
    if y_starts[-1] + TILE_SIZE < pad_h_total:
        y_starts.append(pad_h_total - TILE_SIZE)
    if x_starts[-1] + TILE_SIZE < pad_w_total:
        x_starts.append(pad_w_total - TILE_SIZE)

    tile_coords = []
    tiles_np    = []
    for y0 in y_starts:
        for x0 in x_starts:
            tile_coords.append((y0, x0, y0 + TILE_SIZE, x0 + TILE_SIZE))
            tiles_np.append(padded_img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])

    total_sw_tiles = len(tile_coords)

    # ── Mini-batch forward passes ──────────────────────────────────────────
    all_pred_indices = []
    all_pred_vecs    = []
    all_conv_outputs = []

    for batch_start in range(0, total_sw_tiles, MINI_BATCH_SIZE):
        batch_end   = min(batch_start + MINI_BATCH_SIZE, total_sw_tiles)
        batch_tiles = tiles_np[batch_start:batch_end]

        if progress_bar is not None:
            progress_bar.progress(
                batch_start / total_sw_tiles,
                text=f"Predicting tiles {batch_start + 1}\u2013{batch_end} of {total_sw_tiles} \u2026"
            )

        b_pred_indices, b_pred_vecs, b_conv_outputs = predict_tiles_batch(
            batch_tiles, sensitivity
        )
        all_pred_indices.extend(b_pred_indices)
        all_pred_vecs.append(b_pred_vecs)
        all_conv_outputs.append(b_conv_outputs)

    pred_indices = all_pred_indices
    pred_vecs    = np.concatenate(all_pred_vecs,    axis=0)
    conv_outputs = np.concatenate(all_conv_outputs, axis=0)

    # ── Score-map fusion ───────────────────────────────────────────────────
    # For every pixel, accumulate P(cracked) from all tiles that cover it,
    # then divide by the number of tiles covering that pixel.
    # This gives a smooth, averaged crack probability map at full resolution.
    crack_score_map = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)
    count_map       = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)

    for tile_idx, (y0, x0, y1, x1) in enumerate(tile_coords):
        p_cracked = float(pred_vecs[tile_idx][1])   # P(cracked) for this tile
        crack_score_map[y0:y1, x0:x1] += p_cracked
        count_map      [y0:y1, x0:x1] += 1.0

    count_map       = np.maximum(count_map, 1)
    crack_score_map = crack_score_map / count_map   # averaged probability map
    crack_score_map = crack_score_map[:orig_h, :orig_w]   # crop to original size

    # ── Build non-overlapping display grid ────────────────────────────────
    # The UI shows a clean grid; each cell's label comes from the fused score map.
    n_rows = math.ceil(orig_h / TILE_SIZE)
    n_cols = math.ceil(orig_w  / TILE_SIZE)

    CLASS_COLORS_BGR = {
        0: (0,   200,  0),    # Normal   → green
        1: (0,   0,   255),   # Cracked  → red
        2: (0,   165, 255),   # Not wall → orange
    }
    CLASS_LABELS = {0: "Normal", 1: "Cracked", 2: "Not a Wall"}

    # Pad to exact grid size for drawing
    pad_grid_h = n_rows * TILE_SIZE - orig_h
    pad_grid_w = n_cols * TILE_SIZE - orig_w
    padded_grid = cv2.copyMakeBorder(original_img, 0, pad_grid_h, 0, pad_grid_w,
                                     cv2.BORDER_REFLECT)

    # Extend score map to grid size
    crack_score_full = np.zeros((n_rows * TILE_SIZE, n_cols * TILE_SIZE), dtype=np.float32)
    crack_score_full[:orig_h, :orig_w] = crack_score_map

    output_canvas        = padded_grid.copy()
    tile_grid_overlay    = padded_grid.copy().astype(np.float32)
    contours_only_canvas = padded_grid.copy()

    tile_results  = []
    cracked_count = 0

    for r in range(n_rows):
        for c in range(n_cols):
            gy0, gy1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            gx0, gx1 = c * TILE_SIZE, (c + 1) * TILE_SIZE

            # Average crack probability in this grid cell from the fused map
            cell_score = float(crack_score_full[gy0:gy1, gx0:gx1].mean())
            conf_pct   = cell_score * 100.0

            # Classify the cell
            if conf_pct >= CONF_THRESHOLD:
                pred_index = 1   # Cracked
            else:
                # Check if "Not a Wall" using the closest sliding-window tile
                cy, cx = (gy0 + gy1) // 2, (gx0 + gx1) // 2
                best_idx, best_dist = 0, float("inf")
                for ti, (ty0, tx0, ty1, tx1) in enumerate(tile_coords):
                    d = abs((ty0 + ty1) // 2 - cy) + abs((tx0 + tx1) // 2 - cx)
                    if d < best_dist:
                        best_dist, best_idx = d, ti
                not_wall_prob = float(pred_vecs[best_idx][2])
                pred_index = 2 if not_wall_prob > 0.5 else 0

            tile_results.append({
                "row":        r,
                "col":        c,
                "pred":       pred_index,
                "label":      CLASS_LABELS[pred_index],
                "confidence": round(conf_pct, 2),
            })

            color_bgr = CLASS_COLORS_BGR[pred_index]
            color_rgb = color_bgr[::-1]
            alpha     = 0.35

            # Colour the grid overlay
            tile_grid_overlay[gy0:gy1, gx0:gx1] = (
                (1 - alpha) * tile_grid_overlay[gy0:gy1, gx0:gx1].astype(np.float32)
                + alpha * np.array(color_rgb, dtype=np.float32)
            )
            cv2.rectangle(output_canvas,   (gx0, gy0), (gx1-1, gy1-1), color_rgb, 2)
            cv2.rectangle(tile_grid_overlay.astype(np.uint8),
                          (gx0, gy0), (gx1-1, gy1-1), color_rgb, 2)

            # Draw contours in cracked cells using the fused score map
            if pred_index == 1:
                cracked_count += 1
                cell_heat = crack_score_full[gy0:gy1, gx0:gx1].copy()
                cell_heat = np.maximum(cell_heat, 0)
                if cell_heat.max() > 0:
                    cell_heat = cell_heat / cell_heat.max()
                cell_uint8    = np.uint8(255 * cell_heat)
                _, thresh_map = cv2.threshold(cell_uint8, int(255 * 0.5),
                                              255, cv2.THRESH_BINARY)
                contours_c, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                shifted = [cnt + np.array([[[gx0, gy0]]]) for cnt in contours_c]
                cv2.drawContours(output_canvas,        shifted, -1, (255, 0, 0), contour_thickness)
                cv2.drawContours(contours_only_canvas, shifted, -1, (255, 0, 0), contour_thickness)

        if progress_bar is not None:
            progress_bar.progress(
                (r + 1) / n_rows,
                text=f"Post-processing row {r + 1}/{n_rows} \u2026"
            )

    # ── Numbered tile grid ─────────────────────────────────────────────────
    numbered_canvas = padded_grid.copy()
    font            = cv2.FONT_HERSHEY_SIMPLEX
    max_img_dim     = max(orig_w, orig_h)
    font_scale      = max(0.4, max_img_dim / 1500)
    font_thickness  = max(1, int(font_scale * 2.5))
    tile_number     = 0
    result_lookup   = {(t["row"], t["col"]): t for t in tile_results}

    for r2 in range(n_rows):
        for c2 in range(n_cols):
            y0t   = r2 * TILE_SIZE
            x0t   = c2 * TILE_SIZE
            t_info    = result_lookup[(r2, c2)]
            color_bgr = CLASS_COLORS_BGR[t_info["pred"]]
            color_rgb = color_bgr[::-1]
            cv2.rectangle(numbered_canvas,
                          (x0t, y0t), (x0t + TILE_SIZE - 1, y0t + TILE_SIZE - 1),
                          color_rgb, 2)
            label_str   = str(tile_number)
            padding     = max(4, int(TILE_SIZE * 0.04))
            (tw, th), _ = cv2.getTextSize(label_str, font, font_scale, font_thickness)
            tx = x0t + padding
            ty = y0t + th + padding
            cv2.putText(numbered_canvas, label_str, (tx+1, ty+1),
                        font, font_scale, (255, 255, 255), font_thickness+1, cv2.LINE_AA)
            cv2.putText(numbered_canvas, label_str, (tx, ty),
                        font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            tile_number += 1

    # ── Crop all outputs to original dimensions ────────────────────────────
    result_image        = Image.fromarray(output_canvas[:orig_h, :orig_w])
    tile_grid_image     = Image.fromarray(tile_grid_overlay.astype(np.uint8)[:orig_h, :orig_w])
    contours_only_image = Image.fromarray(contours_only_canvas[:orig_h, :orig_w])
    numbered_image      = Image.fromarray(numbered_canvas[:orig_h, :orig_w])

    summary = {
        "total":    n_rows * n_cols,
        "cracked":  cracked_count,
        "normal":   sum(1 for t in tile_results if t["pred"] == 0),
        "not_wall": sum(1 for t in tile_results if t["pred"] == 2),
        "tiles":    tile_results,
        "grid":     (n_rows, n_cols),
        "overlap":  overlap,
        "step_px":  step,
        "sw_tiles": total_sw_tiles,
    }
    return result_image, tile_grid_image, contours_only_image, numbered_image, summary


# ══════════════════════════════════════════════
# Main Streamlit UI
# ══════════════════════════════════════════════

if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            image = Image.open(file)
            if image is None:
                raise ValueError("Uploaded file is not a valid image.")
            image = correct_orientation(image)

            # ── Auto-resize large images to protect memory ─────────────
            MAX_DIMENSION = 3000
            orig_w, orig_h = image.size
            if max(orig_w, orig_h) > MAX_DIMENSION:
                scale  = MAX_DIMENSION / max(orig_w, orig_h)
                new_w  = int(orig_w * scale)
                new_h  = int(orig_h * scale)
                image  = image.resize((new_w, new_h), Image.LANCZOS)
                st.info(
                    f"\U0001f4d0 Image resized from {orig_w}\xd7{orig_h} to {new_w}\xd7{new_h} px "
                    f"to fit within memory limits. Detection quality is not affected."
                )

            # ── Whole-image prediction ─────────────────────────────────
            predictions, image_with_border, contours_with_border, \
                heatmap_image, contoured_image, overlay_img = import_and_predict(image)

            if predictions is not None:
                predicted_class        = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                if predicted_class == 0:
                    st.success("\u2705 This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error("\u274c This wall is a cracked brick wall.")
                elif predicted_class == 2:
                    st.warning("\u26a0\ufe0f This is not a brick wall.")
                else:
                    st.error(f"\u2753 Unknown prediction result: {predicted_class}")

                st.write("**Prediction Percentages:**")
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 14px;
                                color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
                        <div style="text-align: center; flex: 1;">
                            \U0001f7e2 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%
                        </div>
                        <div style="text-align: center; flex: 1;">
                            \U0001f534 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%
                        </div>
                        <div style="text-align: center; flex: 1;">
                            \U0001f7e0 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")

                with st.expander("\U0001f50d Sensitivity Settings"):
                    sensitivity = st.slider(
                        "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                        min_value=0, max_value=12, value=9, step=1, format="%.1f",
                    )

                predictions, image_with_border, contours_with_border, \
                    heatmap_image, contoured_image, overlay_img = import_and_predict(
                        image, sensitivity=sensitivity
                    )

                st.subheader("\U0001f50e Whole-Image Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col2:
                    if predicted_class == 1:
                        st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
                    else:
                        st.image(image,
                                 caption="No cracks detected" if predicted_class == 0 else "No wall detected",
                                 use_container_width=True)
                with col3:
                    if predicted_class == 1:
                        st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
                    else:
                        st.image(image,
                                 caption="No cracks detected" if predicted_class == 0 else "No wall detected",
                                 use_container_width=True)
                with col4:
                    if predicted_class == 1:
                        st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
                    else:
                        st.image(image,
                                 caption="No cracks detected" if predicted_class == 0 else "No wall detected",
                                 use_container_width=True)

                image_with_border    = add_canvas(image_with_border)
                contours_with_border = add_canvas(contours_with_border)
                st.write("")
                if st.checkbox("Original vs Cracked Slider"):
                    center_style = """
                    <style>
                    .centered-image-container { display: flex; justify-content: center; align-items: center; }
                    </style>"""
                    st.markdown(center_style, unsafe_allow_html=True)
                    st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
                    if predicted_class == 1:
                        image_comparison(img1=image_with_border, img2=contours_with_border,
                                         label1="Uploaded Image", label2="Cracks Localization",
                                         show_labels=False)
                    else:
                        image_comparison(img1=image_with_border, img2=image_with_border,
                                         label1="Uploaded Image", label2="Cracks Localization",
                                         show_labels=False)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════════
                # Tile-based section — sliding window
                # ══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("\U0001f9e9 Tile-Based Segment Analysis — Sliding Window")
                st.caption(
                    "The image is scanned with overlapping 224 \xd7 224 px tiles. "
                    "Overlapping tiles average their predictions so cracks on tile "
                    "borders are never missed. "
                    "\U0001f7e2 Green\u202f=\u202fNormal  |  \U0001f534 Red\u202f=\u202fCracked  |  "
                    "\U0001f7e0 Orange\u202f=\u202fNot a wall"
                )

                with st.expander("\U0001fa9f Sliding Window Settings"):
                    overlap = st.slider(
                        "Tile Overlap — higher values catch more border cracks but process more tiles",
                        min_value=0.0, max_value=0.75, value=0.5, step=0.25,
                        format="%.2f",
                        help="0.0\u202f=\u202fno overlap  |  0.5\u202f=\u202f50% overlap  |  0.75\u202f=\u202f75% overlap"
                    )
                    step_px     = max(1, int(224 * (1.0 - overlap)))
                    est_w, est_h = image.size
                    est_tiles   = (math.ceil(est_w / step_px) + 1) * (math.ceil(est_h / step_px) + 1)
                    st.caption(
                        f"Step size: **{step_px} px** — "
                        f"estimated sliding-window tiles: **~{est_tiles}** "
                        f"(display grid: {math.ceil(est_h/224)} \xd7 {math.ceil(est_w/224)} cells)"
                    )

                run_tiled = st.button("\u25b6 Run Tile-Based Analysis", type="primary")

                if run_tiled:
                    progress_bar = st.progress(0, text="Starting tile analysis \u2026")
                    tiled_result, tile_grid_img, contours_only_img, numbered_img, summary = tiled_crack_detection(
                        image, sensitivity=sensitivity, progress_bar=progress_bar,
                        overlap=overlap
                    )
                    progress_bar.empty()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Display Grid Tiles", summary["total"])
                    m2.metric("\U0001f534 Cracked",   summary["cracked"],
                              delta=f"{summary['cracked']/summary['total']*100:.1f}%",
                              delta_color="inverse")
                    m3.metric("\U0001f7e2 Normal",    summary["normal"])
                    m4.metric("\U0001f7e0 Not a Wall",summary["not_wall"])

                    sw_info = (
                        f"Overlap: **{int(summary['overlap']*100)}%** — "
                        f"step: **{summary['step_px']} px** — "
                        f"sliding-window tiles processed: **{summary['sw_tiles']}**"
                    )
                    st.caption(sw_info)

                    cracked_pct = summary["cracked"] / summary["total"] * 100
                    if summary["cracked"] == 0:
                        st.success("\u2705 No cracked tiles detected across the entire image.")
                    elif cracked_pct < 25:
                        st.warning(f"\u26a0\ufe0f Minor cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
                    elif cracked_pct < 60:
                        st.error(f"\u274c Moderate cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
                    else:
                        st.error(f"\U0001f6a8 Severe cracking detected: {cracked_pct:.1f}% of tiles are cracked.")

                    st.write("")
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.image(tile_grid_img,
                                 caption="Tile Grid — colour-coded by class",
                                 use_container_width=True)
                    with tc2:
                        st.image(tiled_result,
                                 caption="Contour lines drawn in cracked tiles",
                                 use_container_width=True)

                    st.write("")
                    tc3, tc4 = st.columns(2)
                    with tc3:
                        st.image(contours_only_img,
                                 caption="Crack contours on original image (no tinting)",
                                 use_container_width=True)
                    with tc4:
                        st.image(numbered_img,
                                 caption="Numbered tiles (colour = predicted class)",
                                 use_container_width=True)

                    with st.expander("\U0001f4cb Tile-by-Tile Results"):
                        import pandas as pd
                        df = pd.DataFrame(summary["tiles"])
                        df.columns = ["Row", "Col", "Pred Index", "Label", "Confidence (%)"]
                        df["Confidence (%)"] = df["Confidence (%)"].round(2)
                        def highlight_cracked(row):
                            if row["Label"] == "Cracked":
                                return ["background-color: #ffe0e0"] * len(row)
                            return [""] * len(row)
                        st.dataframe(df.style.apply(highlight_cracked, axis=1),
                                     use_container_width=True)

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
# #____________________________________________________________________________________________________________________________________________________________________________________
# # OLD WORKING CODE
# #____________________________________________________________________________________________________________________________________________________________________________________
# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps, ExifTags, ImageEnhance
# import numpy as np
# import cv2
# from keras.models import Model
# from streamlit_image_comparison import image_comparison
# import math
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

# @st.cache_resource
# #__________________________________________________________________________________________________________________________________________________________________________________
# #For single model selection
# def load_model():
#     try:
#         model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None

# model = load_model()

# # Main area for image upload
# file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# # Function to correct image orientation based on EXIF data
# def correct_orientation(image):
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = image._getexif()
#         if exif is not None:
#             orientation = exif.get(orientation, 1)
#             if orientation == 3:
#                 image = image.rotate(180, expand=True)
#             elif orientation == 6:
#                 image = image.rotate(270, expand=True)
#             elif orientation == 8:
#                 image = image.rotate(90, expand=True)
#     except (AttributeError, KeyError, IndexError):
#         pass
#     return image
    
# # Adding Canvas Background
# def add_canvas(image, fill_color=(255, 255, 255)):
#     """Automatically adjusts canvas size according to image size, with added padding and centers the image on the canvas."""
#     # Get the original image size
#     image_width, image_height = image.size
    
#     # Calculate new canvas size with padding
#     canvas_width = image_width + math.ceil(0.015 * image_width)
#     canvas_height = image_height + math.ceil(0.07 * image_height)
    
#     # Create a new image (canvas) with the calculated size
#     canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    
#     # Calculate the position to paste the image at the center of the canvas
#     paste_position = (
#         (canvas_width - image_width) // 2,
#         (canvas_height - image_height) // 7
#     )
    
#     # Paste the original image onto the canvas
#     canvas.paste(image, paste_position)
    
#     return canvas


# # Function to localize the crack and to make predictions using the TensorFlow model
# def import_and_predict(image_data, sensitivity=9):
#     try:
#         # Convert image to numpy array
#         original_img = np.array(image_data)

#         # Convert image to RGB if it has an alpha channel
#         if original_img.shape[-1] == 4:  # Check if the image has 4 channels
#             original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        
#         # Save original dimensions
#         orig_height, orig_width, _ = original_img.shape
        
#         # Calculate the maximum dimension of the original image
#         max_dimension = max(orig_width, orig_height)
        
#         # Set the scaling factor for contour line thickness based on the max dimension
#         contour_thickness = max(2, int(max_dimension / 200))  # Adjust the divisor to control scaling

#         # Preprocess the image for the model
#         img_resized = cv2.resize(original_img, (224, 224))
#         img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
#         preprocessed_img = img_tensor
        
#         # Define a new model that outputs the conv2d_3 feature maps and the prediction
#         custom_model = Model(inputs=model.inputs, 
#                              outputs=(model.layers[sensitivity].output, model.layers[-1].output))  # `conv2d_3` and predictions

#         # Get the conv2d_3 output and the predictions
#         conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
#         conv2d_3_output = np.squeeze(conv2d_3_output)  # (28, 28, 32) feature maps

#         # Prediction for the image
#         pred = np.argmax(pred_vec)
        
#         # Resize the conv2d_3 output to match the input image size
#         heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

#         # Average all the filters from conv2d_3 to get a single activation map
#         heat_map = np.mean(heat_map_resized, axis=-1)  # (orig_height, orig_width)

#         # Normalize the heatmap for better visualization
#         heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
#         heat_map = heat_map / heat_map.max()  # Normalize to 0-1

#         # Threshold the heatmap to get the regions with the highest activation
#         threshold = 0.5  # Adjust this threshold if needed
#         heat_map_thresh = np.uint8(255 * heat_map)
#         _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

#         # Find contours in the thresholded heatmap
#         contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Convert heatmap to RGB for display
#         heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        
#         # Convert heatmap to PIL format
#         heatmap_image = Image.fromarray(heatmap_colored)
        
#         # Create contoured image
#         contoured_img = original_img.copy()  # Copy original image
#         cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)  # Draw blue contours
        
#         # Convert contoured image to PIL format
#         contoured_image = Image.fromarray(contoured_img)

#         # Overlay heatmap on original image
#         heatmap_image_rgba = heatmap_image.convert("RGBA")
#         original_img_pil = Image.fromarray(original_img).convert("RGBA")
#         heatmap_overlay = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

#         # Draw contours on the heatmap-overlayed image
#         # Convert heatmap-overlayed image to RGB for contour drawing
#         heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
#         heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
#         # heatmap_overlay_np = np.array(heatmap_overlay)
#         cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)  # Draw blue contours

#         # Convert overlay image to PIL format
#         overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

#         # Get the predicted class name
#         class_labels = ["Normal", "Cracked", "Not a Wall"]
#         predicted_class = class_labels[pred]

#         # Add white borders
#         border_size = 10  # Set the border size
#         image_with_border = add_white_border(image_data, border_size)
#         contours_with_border = add_white_border(overlay_img, border_size)

#         return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img 
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")
#         return None, None, None, None, None, None
        
# # Adds border to the image

# def add_white_border(image, border_size):
#     """Add a white border to the image."""
#     return ImageOps.expand(image, border= border_size, fill=(255, 255, 255))

# # Check if a file was uploaded
# if file is None:
#     st.info("Please upload an image file to start the detection.")
# else:
#     with st.spinner("Processing image..."):
#         try:
#             # Try to open the uploaded image using PIL
#             image = Image.open(file)
#             if image is None:
#                 raise ValueError("Uploaded file is not a valid image.")
            
#             # Correct the orientation if necessary
#             image = correct_orientation(image)

            
            
#             # Perform prediction
#             predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image)
#             if predictions is not None:
#                 predicted_class = np.argmax(predictions)
#                 prediction_percentages = predictions[0] * 100

#                  # Display prediction result
#                 if predicted_class == 0:
#                     st.success(f"✅ This is a normal brick wall.")
#                 elif predicted_class == 1:
#                     st.error(f"❌ This wall is a cracked brick wall. ")
#                 elif predicted_class == 2:
#                     st.warning(f"⚠️ This is not a brick wall.")
#                 else:
#                     st.error(f"❓ Unknown prediction result: {predicted_class}")

#                 st.write(f"**Prediction Percentages:**")
#                 # Display predictions in one line
#                 st.markdown(f"""
#                     <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
#                         <div style="text-align: center; flex: 1;">🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%</div>
#                         <div style="text-align: center; flex: 1;">🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%</div>
#                         <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 st.write("")  # Creates a blank line

#                 # st.write("")  # Creates a blank line

#                 # Create an expander for sensitivity adjustment
#                 with st.expander("🔍 Sensitivity Settings"):
#                     # Add a slider for selecting the sensitivity dynamically
#                     sensitivity = st.slider(
#                         "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
#                         min_value=0,   # Minimum value for sensitivity
#                         max_value=12,   # Maximum value for sensitivity
#                         value=9,       # Default value for sensitivity
#                         step=1,        # Step for incremental changes
#                         format="%.1f"    # Format to display sensitivity with one decimal
#                                             )


#                 # Perform prediction again
#                 predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image, sensitivity=sensitivity)

#                 #in one row
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     st.image(image, caption="Uploaded Image", use_container_width=True)
                
#                 with col2:
#                     if predicted_class == 1:
#                         st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
#                     elif predicted_class == 0:
#                         st.image(image, caption="No cracks detected", use_container_width=True)
#                     else:
#                         st.image(image, caption="No wall detected", use_container_width=True)
                        
#                 with col3:
#                     if predicted_class == 1:
#                         st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
#                     elif predicted_class == 0:
#                         st.image(image, caption="No cracks detected", use_container_width=True)
#                     else:
#                         st.image(image, caption="No wall detected", use_container_width=True)
                
#                 with col4:
#                     if predicted_class == 1:
#                         st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
#                     elif predicted_class == 0:
#                         st.image(image, caption="No cracks detected", use_container_width=True)
#                     else:
#                         st.image(image, caption="No wall detected", use_container_width=True)
         
#                 image_with_border = add_canvas(image_with_border)
#                 contours_with_border = add_canvas(contours_with_border)               
#                 st.write("")  # Creates a blank line
#                 if st.checkbox("Original vs Cracked Slider"):
#                     # HTML/CSS for centering the image comparison component
#                     center_style = """
#                     <style>
#                     .centered-image-container {
#                         display: flex;
#                         justify-content: center;
#                         align-items: center;
#                     }
#                     </style>
#                     """
#                     st.markdown(center_style, unsafe_allow_html=True)
                    
#                     # Opening div tag to center the image comparison component
#                     st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
                    
#                     # Conditionally display image comparison
#                     if predicted_class == 1:
#                         image_comparison(
#                             img1=image_with_border, 
#                             img2=contours_with_border,
#                             label1="Uploaded Image",
#                             label2="Cracks Localization",
#                             show_labels=False,
#                         )
#                     else:
#                         image_comparison(
#                             img1=image_with_border, 
#                             img2=image_with_border,
#                             label1="Uploaded Image",
#                             label2="Cracks Localization",
#                             show_labels=False,
#                         )
                
#                     # Closing div tag
#                     st.markdown('</div>', unsafe_allow_html=True)
                            
        
#         except Exception as e:
#             st.error(f"Error processing the uploaded image: {e}")
