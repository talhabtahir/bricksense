# import io
# import math
# import cv2
# import matplotlib.cm as cm
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from keras.models import Model
# from PIL import Image, ImageOps, ExifTags
# from streamlit_image_comparison import image_comparison
# import pandas as pd
# import gc
# import hashlib
# import psutil
# import os

# TILE_SIZE = 224  # Each tile is 224x224 pixels

# # ══════════════════════════════════════════════
# # Resource Monitoring & Cleanup Utilities
# # ══════════════════════════════════════════════

# def get_memory_usage():
#     """Get current memory usage in MB."""
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024


# def aggressive_cleanup():
#     """Aggressively clear memory and caches."""
#     tf.keras.backend.clear_session()
#     gc.collect()
#     import psutil
#     try:
#         import ctypes
#         ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux-specific memory trimming
#     except:
#         pass


# def memory_efficient_image_load(image_bytes, max_dimension=2000):
#     """Load and resize image with minimal memory overhead."""
#     # Load image directly from bytes
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = correct_orientation(image)
    
#     # Immediate resize to reduce memory
#     orig_w, orig_h = image.size
#     if max(orig_w, orig_h) > max_dimension:
#         scale = max_dimension / max(orig_w, orig_h)
#         new_w = int(orig_w * scale)
#         new_h = int(orig_h * scale)
#         image = image.resize((new_w, new_h), Image.LANCZOS)
#         return image, (new_w, new_h), (orig_w, orig_h)
    
#     return image, (orig_w, orig_h), (orig_w, orig_h)


# # ══════════════════════════════════════════════
# # Model loading (cached for the whole session)
# # ══════════════════════════════════════════════

# @st.cache_resource
# def load_model():
#     try:
#         # Load with memory optimization
#         model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
#         # Set model to inference-only to free up some memory
#         model.trainable = False
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None


# @st.cache_resource
# def build_custom_model(sensitivity: int):
#     """Return a two-output Model cached by sensitivity level."""
#     model = load_model()
#     return Model(
#         inputs=model.inputs,
#         outputs=(model.layers[sensitivity].output, model.layers[-1].output),
#     )


# model = load_model()


# # ══════════════════════════════════════════════
# # Helper utilities
# # ══════════════════════════════════════════════

# def correct_orientation(image):
#     """Correct image orientation based on EXIF data."""
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = image._getexif()
#         if exif is not None:
#             orientation_val = exif.get(orientation, 1)
#             if orientation_val == 3:
#                 image = image.rotate(180, expand=True)
#             elif orientation_val == 6:
#                 image = image.rotate(270, expand=True)
#             elif orientation_val == 8:
#                 image = image.rotate(90, expand=True)
#     except (AttributeError, KeyError, IndexError):
#         pass
#     return image


# def add_canvas(image, fill_color=(255, 255, 255)):
#     """Add white canvas around image."""
#     image_width, image_height = image.size
#     canvas_width  = image_width  + math.ceil(0.015 * image_width)
#     canvas_height = image_height + math.ceil(0.07  * image_height)
#     canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
#     paste_position = (
#         (canvas_width  - image_width)  // 2,
#         (canvas_height - image_height) // 7,
#     )
#     canvas.paste(image, paste_position)
#     return canvas


# def add_white_border(image, border_size):
#     """Add white border to image."""
#     return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))


# # ══════════════════════════════════════════════
# # Whole-image prediction (memory-optimized)
# # ══════════════════════════════════════════════

# @st.cache_data(show_spinner=False)
# def import_and_predict(image_bytes: bytes, sensitivity: int = 9):
#     """
#     Optimized prediction with explicit memory management.
#     Returns (pred_vec, image_with_border, contours_with_border,
#              heatmap_image, contoured_image, overlay_img)
#     """
#     try:
#         # Load and convert image to RGB
#         image_data   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         original_img = np.array(image_data, dtype=np.uint8)  # explicit dtype
        
#         orig_height, orig_width, _ = original_img.shape
#         max_dimension     = max(orig_width, orig_height)
#         contour_thickness = max(2, int(max_dimension / 200))

#         # Resize for model prediction
#         img_resized  = cv2.resize(original_img, (224, 224))
#         img_tensor   = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0  # float32, not float16

#         custom_model              = build_custom_model(sensitivity)
#         conv2d_3_output, pred_vec = custom_model.predict(img_tensor, verbose=0)
        
#         # Explicit cleanup of tensor
#         del img_tensor
#         conv2d_3_output           = np.squeeze(conv2d_3_output)

#         # Heatmap processing
#         heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height),
#                                       interpolation=cv2.INTER_LINEAR)
#         del conv2d_3_output  # Free memory
        
#         heat_map = np.mean(heat_map_resized, axis=-1) if heat_map_resized.ndim == 3 else heat_map_resized
#         heat_map = np.maximum(heat_map, 0)
#         if heat_map.max() > 0:
#             heat_map = heat_map / heat_map.max()

#         heat_map_thresh = np.uint8(255 * heat_map)
#         _, thresh_map   = cv2.threshold(heat_map_thresh, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#         contours, _     = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         del heat_map_resized, heat_map_thresh, thresh_map  # Clean up

#         # Heatmap colored
#         heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
#         heatmap_image   = Image.fromarray(heatmap_colored)
#         del heatmap_colored

#         # Draw contours on original
#         contoured_img = original_img.copy()
#         cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
#         contoured_image = Image.fromarray(contoured_img)
#         del contoured_img

#         # Blend heatmap with original
#         heatmap_image_rgba  = heatmap_image.convert("RGBA")
#         original_img_pil    = Image.fromarray(original_img).convert("RGBA")
#         heatmap_overlay     = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

#         heatmap_overlay_rgb    = heatmap_overlay.convert("RGB")
#         heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
#         cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)
#         overlay_img = Image.fromarray(heatmap_overlay_rgb_np)
        
#         del heatmap_image_rgba, original_img_pil, heatmap_overlay, heatmap_overlay_rgb_np

#         # Add borders for display
#         border_size          = 10
#         image_with_border    = add_white_border(Image.fromarray(original_img), border_size)
#         contours_with_border = add_white_border(overlay_img, border_size)
        
#         del original_img, overlay_img  # Clean up original arrays

#         aggressive_cleanup()
#         return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img
        
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")
#         aggressive_cleanup()
#         return None, None, None, None, None, None


# # ══════════════════════════════════════════════
# # Tile-based helpers (memory-optimized)
# # ══════════════════════════════════════════════

# def predict_tiles_batch(tiles_np, sensitivity=9):
#     """Predict batch of tiles with memory optimization."""
#     batch        = np.stack(tiles_np, axis=0).astype(np.float32) / 255.0
#     custom_model = build_custom_model(sensitivity)
#     conv_outputs, pred_vecs = custom_model.predict(batch, verbose=0)
#     pred_indices = [int(np.argmax(pv)) for pv in pred_vecs]
#     del batch  # Explicit cleanup
#     return pred_indices, pred_vecs, conv_outputs


# def ensemble_predict_tiles(tiles_np, sensitivity_levels=(7, 9, 11)):
#     """Soft-voting ensemble with memory optimization."""
#     batch               = np.stack(tiles_np, axis=0).astype(np.float32) / 255.0
#     mid_idx             = len(sensitivity_levels) // 2
#     all_pred_vecs       = []
#     middle_conv_outputs = None

#     for i, sens in enumerate(sensitivity_levels):
#         custom_model             = build_custom_model(sens)
#         conv_outputs, pred_vecs  = custom_model.predict(batch, verbose=0)
#         all_pred_vecs.append(pred_vecs)
#         if i == mid_idx:
#             middle_conv_outputs = conv_outputs.copy()
#         del conv_outputs, pred_vecs  # Explicit cleanup each iteration

#     averaged_pred_vecs = np.mean(np.array(all_pred_vecs), axis=0)
#     pred_indices       = [int(np.argmax(pv)) for pv in averaged_pred_vecs]
#     del batch, all_pred_vecs  # Clean up
#     return pred_indices, averaged_pred_vecs, middle_conv_outputs


# # ══════════════════════════════════════════════
# # Tile-based crack detection (memory-optimized)
# # ══════════════════════════════════════════════

# @st.cache_data(show_spinner=False)
# def tiled_crack_detection(image_bytes: bytes,
#                            sensitivity: int = 9,
#                            confidence_threshold: float = 95.0,
#                            use_ensemble: bool = False,
#                            ensemble_levels: tuple = (7, 9, 11)):
#     """
#     Memory-optimized tile-based detection.
#     """
#     try:
#         image_data   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         original_img = np.array(image_data, dtype=np.uint8)
#         image_data.close()  # Explicitly close PIL image
#         del image_data
        
#         orig_h, orig_w, _ = original_img.shape

#         pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
#         pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE
#         padded_img = cv2.copyMakeBorder(original_img, 0, pad_h, 0, pad_w,
#                                         cv2.BORDER_REFLECT)
#         pad_h_total, pad_w_total = padded_img.shape[:2]

#         n_rows      = pad_h_total // TILE_SIZE
#         n_cols      = pad_w_total // TILE_SIZE
#         total_tiles = n_rows * n_cols

#         contour_thickness = max(2, int(max(orig_w, orig_h) / 200))

#         output_canvas     = padded_img.copy()
#         tile_grid_overlay = padded_img.copy().astype(np.float32)

#         CLASS_COLORS_BGR = {
#             0: (0,   200,  0),
#             1: (0,   0,   255),
#             2: (0,   165, 255),
#         }
#         CLASS_LABELS = {0: "Normal", 1: "Cracked", 2: "Not a Wall"}

#         tile_results    = []
#         cracked_count   = 0
#         MINI_BATCH_SIZE = 4  # Reduced from 8 for memory efficiency

#         # Build tile list
#         tile_coords = []
#         tiles_np    = []
#         for r in range(n_rows):
#             for c in range(n_cols):
#                 y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
#                 x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
#                 tile_coords.append((r, c, y0, y1, x0, x1))
#                 tiles_np.append(padded_img[y0:y1, x0:x1].astype(np.uint8))

#         all_pred_indices = []
#         all_pred_vecs    = []
#         all_conv_outputs = []

#         # Process tiles in mini-batches
#         for batch_start in range(0, total_tiles, MINI_BATCH_SIZE):
#             batch_end   = min(batch_start + MINI_BATCH_SIZE, total_tiles)
#             batch_tiles = tiles_np[batch_start:batch_end]

#             if use_ensemble:
#                 b_pred_indices, b_pred_vecs, b_conv_outputs = ensemble_predict_tiles(
#                     batch_tiles, sensitivity_levels=ensemble_levels
#                 )
#             else:
#                 b_pred_indices, b_pred_vecs, b_conv_outputs = predict_tiles_batch(
#                     batch_tiles, sensitivity
#                 )

#             all_pred_indices.extend(b_pred_indices)
#             all_pred_vecs.append(b_pred_vecs)
#             if b_conv_outputs is not None:
#                 all_conv_outputs.append(b_conv_outputs)

#             del batch_tiles, b_pred_indices, b_pred_vecs, b_conv_outputs

#         del tiles_np  # Free tile list

#         pred_indices = all_pred_indices
#         pred_vecs    = np.concatenate(all_pred_vecs,    axis=0)
#         conv_outputs = np.concatenate(all_conv_outputs, axis=0) if all_conv_outputs else None
        
#         del all_pred_vecs, all_conv_outputs

#         # Process results
#         for tile_idx, (r, c, y0, y1, x0, x1) in enumerate(tile_coords):
#             pred_index  = pred_indices[tile_idx]
#             pred_vec    = pred_vecs[tile_idx]
            
#             conf = float(pred_vec[pred_index]) * 100
#             if pred_index == 1 and conf < confidence_threshold:
#                 pred_index = 0

#             tile_results.append({
#                 "row":        r,
#                 "col":        c,
#                 "pred":       pred_index,
#                 "label":      CLASS_LABELS[pred_index],
#                 "confidence": conf,
#             })

#             color_bgr = CLASS_COLORS_BGR[pred_index]
#             alpha     = 0.35
#             tile_grid_overlay[y0:y1, x0:x1] = (
#                 (1 - alpha) * tile_grid_overlay[y0:y1, x0:x1].astype(np.float32)
#                 + alpha * np.array(color_bgr[::-1], dtype=np.float32)
#             )

#             cv2.rectangle(output_canvas, (x0, y0), (x1 - 1, y1 - 1), color_bgr[::-1], 2)
#             cv2.rectangle(tile_grid_overlay.astype(np.uint8),
#                           (x0, y0), (x1 - 1, y1 - 1), color_bgr[::-1], 2)

#             if pred_index == 1 and conv_outputs is not None:
#                 cracked_count += 1
#                 conv_output = conv_outputs[tile_idx]
#                 heat = np.mean(conv_output, axis=-1) if conv_output.ndim == 3 else conv_output
#                 heat = np.maximum(heat, 0)
#                 if heat.max() > 0:
#                     heat = heat / heat.max()
#                 heat_resized  = cv2.resize(heat, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
#                 heat_uint8    = np.uint8(255 * heat_resized)
#                 _, thresh_map = cv2.threshold(heat_uint8, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#                 contours, _   = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 shifted = [cnt + np.array([[[x0, y0]]]) for cnt in contours]
#                 cv2.drawContours(output_canvas, shifted, -1, (255, 0, 0), contour_thickness)
#                 del heat, heat_resized, heat_uint8, thresh_map

#         # ── Contours-only image ───────────────────────────────────────
#         contours_only_canvas = padded_img.copy()
#         for tile_idx2, (r2, c2, y0t, y1t, x0t, x1t) in enumerate(tile_coords):
#             t = tile_results[tile_idx2]
#             if t["pred"] == 1 and conv_outputs is not None:
#                 conv_out2 = conv_outputs[tile_idx2]
#                 heat2     = np.mean(conv_out2, axis=-1) if conv_out2.ndim == 3 else conv_out2
#                 heat2     = np.maximum(heat2, 0)
#                 if heat2.max() > 0:
#                     heat2 = heat2 / heat2.max()
#                 heat2_resized = cv2.resize(heat2, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
#                 heat2_uint8   = np.uint8(255 * heat2_resized)
#                 _, thresh2    = cv2.threshold(heat2_uint8, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#                 contours2, _  = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 shifted2 = [cnt + np.array([[[x0t, y0t]]]) for cnt in contours2]
#                 cv2.drawContours(contours_only_canvas, shifted2, -1, (255, 0, 0), contour_thickness)
#                 del heat2, heat2_resized, heat2_uint8, thresh2

#         contours_only_image = Image.fromarray(contours_only_canvas[:orig_h, :orig_w])

#         # ── Numbered tile grid ────────────────────────────────────────
#         numbered_canvas = padded_img.copy()
#         font            = cv2.FONT_HERSHEY_SIMPLEX
#         tile_number     = 0
#         max_img_dim     = max(orig_w, orig_h)
#         font_scale      = max(0.4, max_img_dim / 1500)
#         font_thickness  = max(1, int(font_scale * 2.5))
        
#         for r2 in range(n_rows):
#             for c2 in range(n_cols):
#                 y0t    = r2 * TILE_SIZE
#                 x0t    = c2 * TILE_SIZE
#                 t_info = next(t for t in tile_results if t["row"] == r2 and t["col"] == c2)
#                 color_bgr = CLASS_COLORS_BGR[t_info["pred"]]
#                 color_rgb = color_bgr[::-1]
#                 cv2.rectangle(numbered_canvas,
#                               (x0t, y0t), (x0t + TILE_SIZE - 1, y0t + TILE_SIZE - 1),
#                               color_rgb, 2)
#                 label_str   = str(tile_number)
#                 padding     = max(4, int(TILE_SIZE * 0.04))
#                 (tw, th), _ = cv2.getTextSize(label_str, font, font_scale, font_thickness)
#                 tx = x0t + padding
#                 ty = y0t + th + padding
#                 cv2.putText(numbered_canvas, label_str, (tx + 1, ty + 1),
#                             font, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
#                 cv2.putText(numbered_canvas, label_str, (tx, ty),
#                             font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#                 tile_number += 1

#         numbered_image = Image.fromarray(numbered_canvas[:orig_h, :orig_w])

#         result_image    = Image.fromarray(output_canvas[:orig_h, :orig_w])
#         tile_grid_image = Image.fromarray(tile_grid_overlay.astype(np.uint8)[:orig_h, :orig_w])

#         # Clean up large arrays
#         del padded_img, output_canvas, tile_grid_overlay, contours_only_canvas, numbered_canvas
#         del pred_vecs, conv_outputs

#         summary = {
#             "total":    total_tiles,
#             "cracked":  cracked_count,
#             "normal":   sum(1 for t in tile_results if t["pred"] == 0),
#             "not_wall": sum(1 for t in tile_results if t["pred"] == 2),
#             "tiles":    tile_results,
#             "grid":     (n_rows, n_cols),
#         }
        
#         aggressive_cleanup()
#         return result_image, tile_grid_image, contours_only_image, numbered_image, summary
        
#     except Exception as e:
#         st.error(f"Error in tiled detection: {e}")
#         aggressive_cleanup()
#         return None, None, None, None, None


# # ══════════════════════════════════════════════
# # Session-state initialisation
# # ══════════════════════════════════════════════

# def init_session_state():
#     defaults = {
#         "whole_image_results":  None,
#         "tile_results":         None,
#         "last_image_hash":      None,
#         "run_sensitivity":      None,
#         "run_confidence":       None,
#         "run_ensemble":         None,
#         "run_ensemble_lvls":    None,
#         "memory_warning_shown": False,
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# init_session_state()


# # ══════════════════════════════════════════════
# # Main Streamlit UI
# # ══════════════════════════════════════════════

# st.set_page_config(page_title="Crack Detection", layout="wide", initial_sidebar_state="expanded")

# # Memory monitoring in sidebar
# with st.sidebar:
#     st.subheader("📊 System Monitor")
#     memory_mb = get_memory_usage()
#     st.metric("Memory Usage", f"{memory_mb:.1f} MB")
#     if memory_mb > 1500:
#         st.warning("⚠️ High memory usage detected. Consider restarting or closing unused tabs.")
#     if st.button("🗑️ Force Cleanup"):
#         aggressive_cleanup()
#         st.success("Memory cleaned!")
#         st.rerun()

# file = st.file_uploader(
#     "Please upload an image of the brick wall",
#     type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"),
# )

# if file is None:
#     st.info("Please upload an image file to start the detection.")
# else:
#     image_bytes = file.getvalue()
#     image_hash = hashlib.md5(image_bytes).hexdigest()

#     # Reset on new image
#     if st.session_state["last_image_hash"] != image_hash:
#         st.session_state["whole_image_results"] = None
#         st.session_state["tile_results"]        = None
#         st.session_state["run_sensitivity"]     = None
#         st.session_state["run_confidence"]      = None
#         st.session_state["run_ensemble"]        = None
#         st.session_state["run_ensemble_lvls"]   = None
#         st.session_state["last_image_hash"]     = image_hash
#         aggressive_cleanup()

#     try:
#         image, (new_w, new_h), (orig_w, orig_h) = memory_efficient_image_load(image_bytes, max_dimension=2000)
        
#         if image is None:
#             raise ValueError("Uploaded file is not a valid image.")

#         if (new_w, new_h) != (orig_w, orig_h):
#             st.info(
#                 f"📐 Image resized from {orig_w}×{orig_h} to {new_w}×{new_h} px "
#                 f"to fit within memory limits."
#             )
#             buf = io.BytesIO()
#             image.save(buf, format="PNG")
#             image_bytes = buf.getvalue()
#             buf.close()
#             del buf

#         # ═══════════════════════════════════════════════════════════
#         # Settings expander
#         # ═══════════════════════════════════════════════════════════
#         with st.expander("🔍 Sensitivity Settings"):
#             sensitivity = st.slider(
#                 "Adjust Detection Sensitivity (Higher = more sensitive)",
#                 min_value=0, max_value=12, value=9, step=1,
#             )
#             st.divider()
#             confidence_threshold = st.slider(
#                 "🎚️ Crack Confidence Threshold (%)",
#                 min_value=10.0, max_value=99.0, value=95.0, step=1.0,
#                 help=(
#                     "Tiles predicted as 'Cracked' below this confidence are "
#                     "reclassified as Normal. Lower = more tiles flagged."
#                 ),
#             )
#             st.divider()
#             use_ensemble = st.toggle(
#                 "🧪 Ensemble Mode (average predictions across multiple sensitivity levels)",
#                 value=False,
#             )
#             if use_ensemble:
#                 ensemble_levels = tuple(sorted(st.multiselect(
#                     "Sensitivity levels to ensemble",
#                     options=list(range(0, 13)),
#                     default=[7, 9, 11],
#                     help="Select at least 2 levels.",
#                 )))
#                 if len(ensemble_levels) < 2:
#                     st.warning("⚠️ Please select at least 2 sensitivity levels.")
#                     use_ensemble   = False
#                     ensemble_levels = ()
#             else:
#                 ensemble_levels = ()

#         # ═══════════════════════════════════════════════════════════
#         # Whole-image analysis
#         # ═══════════════════════════════════════════════════════════
#         run_whole = st.button("🔬 Run / Refresh Whole-Image Analysis", type="secondary")

#         if run_whole:
#             with st.spinner("Running whole-image analysis…"):
#                 results = import_and_predict(image_bytes, sensitivity)
#             st.session_state["whole_image_results"] = results
#             st.session_state["run_sensitivity"]     = sensitivity
#             st.session_state["run_ensemble"]        = use_ensemble
#             st.session_state["run_ensemble_lvls"]   = ensemble_levels

#         # Stale warning for whole-image
#         if st.session_state["whole_image_results"] is not None:
#             whole_stale = (
#                 st.session_state["run_sensitivity"] != sensitivity
#                 or st.session_state["run_ensemble"] != use_ensemble
#                 or (use_ensemble and st.session_state["run_ensemble_lvls"] != ensemble_levels)
#             )
#             if whole_stale:
#                 st.warning(
#                     "⚠️ Settings have changed since the last whole-image analysis. "
#                     "Click **🔬 Run / Refresh Whole-Image Analysis** to update."
#                 )

#         if st.session_state["whole_image_results"] is None:
#             st.info("Click **🔬 Run / Refresh Whole-Image Analysis** to analyse the uploaded image.")
#         else:
#             (predictions, image_with_border, contours_with_border,
#              heatmap_image, contoured_image, overlay_img) = st.session_state["whole_image_results"]

#             if predictions is not None:
#                 predicted_class        = np.argmax(predictions)
#                 prediction_percentages = predictions[0] * 100

#                 if predicted_class == 0:
#                     st.success("✅ This is a normal brick wall.")
#                 elif predicted_class == 1:
#                     st.error("❌ This wall is a cracked brick wall.")
#                 elif predicted_class == 2:
#                     st.warning("⚠️ This is not a brick wall.")
#                 else:
#                     st.error(f"❓ Unknown prediction result: {predicted_class}")

#                 st.write("**Prediction Percentages:**")
#                 st.markdown(f"""
#                     <div style="display:flex;justify-content:space-between;font-size:14px;
#                                 color:#e0e0e0;background-color:#808080;padding:3px;border-radius:9px;">
#                         <div style="text-align:center;flex:1;">
#                             🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%
#                         </div>
#                         <div style="text-align:center;flex:1;">
#                             🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%
#                         </div>
#                         <div style="text-align:center;flex:1;">
#                             🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 st.write("")
#                 st.subheader("🔎 Whole-Image Analysis")
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.image(image, caption="Uploaded Image", use_container_width=True)
#                 with col2:
#                     if predicted_class == 1:
#                         st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)
#                 with col3:
#                     if predicted_class == 1:
#                         st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)
#                 with col4:
#                     if predicted_class == 1:
#                         st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)

#                 # Before / after slider
#                 image_with_border    = add_canvas(image_with_border)
#                 contours_with_border = add_canvas(contours_with_border)
#                 st.write("")
#                 if st.checkbox("Original vs Cracked Slider"):
#                     st.markdown(
#                         "<style>.centered-image-container{display:flex;justify-content:center;}"
#                         "</style><div class='centered-image-container'>",
#                         unsafe_allow_html=True,
#                     )
#                     if predicted_class == 1:
#                         image_comparison(
#                             img1=image_with_border, img2=contours_with_border,
#                             label1="Uploaded Image", label2="Cracks Localization",
#                             show_labels=False,
#                         )
#                     else:
#                         image_comparison(
#                             img1=image_with_border, img2=image_with_border,
#                             label1="Uploaded Image", label2="Cracks Localization",
#                             show_labels=False,
#                         )
#                     st.markdown("</div>", unsafe_allow_html=True)

#         # ═══════════════════════════════════════════════════════════
#         # Tile-based section
#         # ═══════════════════════════════════════════════════════════
#         st.divider()
#         st.subheader("🧩 Tile-Based Segment Analysis (224 × 224 px tiles)")
#         st.caption(
#             "The image is divided into 224 × 224 pixel tiles. "
#             "Each tile is independently classified. "
#             "🟢 Green = Normal  |  🔴 Red = Cracked  |  🟠 Orange = Not a wall"
#         )

#         run_tiled = st.button("▶ Run Tile-Based Analysis", type="primary")

#         if run_tiled:
#             if use_ensemble:
#                 st.info(
#                     f"🧪 Ensemble mode active — averaging predictions across "
#                     f"sensitivity levels: {sorted(ensemble_levels)}"
#                 )
#             with st.spinner("Running tile analysis … this may take a moment."):
#                 tile_output = tiled_crack_detection(
#                     image_bytes,
#                     sensitivity=sensitivity,
#                     confidence_threshold=confidence_threshold,
#                     use_ensemble=use_ensemble,
#                     ensemble_levels=ensemble_levels,
#                 )
#             st.session_state["tile_results"]      = tile_output
#             st.session_state["run_sensitivity"]   = sensitivity
#             st.session_state["run_confidence"]    = confidence_threshold
#             st.session_state["run_ensemble"]      = use_ensemble
#             st.session_state["run_ensemble_lvls"] = ensemble_levels

#         # Stale warning for tile results
#         if st.session_state["tile_results"] is not None:
#             tile_stale = (
#                 st.session_state["run_sensitivity"] != sensitivity
#                 or st.session_state["run_confidence"] != confidence_threshold
#                 or st.session_state["run_ensemble"]   != use_ensemble
#                 or (use_ensemble and st.session_state["run_ensemble_lvls"] != ensemble_levels)
#             )
#             if tile_stale:
#                 st.warning(
#                     "⚠️ Settings have changed since the last tile analysis. "
#                     "Click **▶ Run Tile-Based Analysis** to update the results."
#                 )

#         if st.session_state["tile_results"] is None:
#             st.info("Click **▶ Run Tile-Based Analysis** to run tile-based crack detection.")
#         else:
#             tiled_result, tile_grid_img, contours_only_img, numbered_img, summary = \
#                 st.session_state["tile_results"]

#             if summary is not None:
#                 m1, m2, m3, m4 = st.columns(4)
#                 m1.metric("Total Tiles",   summary["total"])
#                 m2.metric("🔴 Cracked",    summary["cracked"],
#                           delta=f"{summary['cracked'] / summary['total'] * 100:.1f}%",
#                           delta_color="inverse")
#                 m3.metric("🟢 Normal",     summary["normal"])
#                 m4.metric("🟠 Not a Wall", summary["not_wall"])

#                 cracked_pct = summary["cracked"] / summary["total"] * 100
#                 if summary["cracked"] == 0:
#                     st.success("✅ No cracked tiles detected across the entire image.")
#                 elif cracked_pct < 25:
#                     st.warning(f"⚠️ Minor cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
#                 elif cracked_pct < 60:
#                     st.error(f"❌ Moderate cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
#                 else:
#                     st.error(f"🚨 Severe cracking detected: {cracked_pct:.1f}% of tiles are cracked.")

#                 st.write("")
#                 tc1, tc2 = st.columns(2)
#                 with tc1:
#                     st.image(tile_grid_img,
#                              caption="Tile Grid — colour-coded by class",
#                              use_container_width=True)
#                 with tc2:
#                     st.image(tiled_result,
#                              caption="Contour lines drawn in cracked tiles",
#                              use_container_width=True)

#                 st.write("")
#                 tc3, tc4 = st.columns(2)
#                 with tc3:
#                     st.image(contours_only_img,
#                              caption="Crack contours on original image (no tinting)",
#                              use_container_width=True)
#                 with tc4:
#                     st.image(numbered_img,
#                              caption="Numbered tiles (colour = predicted class)",
#                              use_container_width=True)

#                 with st.expander("📋 Tile-by-Tile Results"):
#                     df = pd.DataFrame(summary["tiles"])
#                     df.columns = ["Row", "Col", "Pred Index", "Label", "Confidence (%)"]
#                     df["Confidence (%)"] = df["Confidence (%)"].round(2)

#                     def highlight_cracked(row):
#                         if row["Label"] == "Cracked":
#                             return ["background-color: #ffe0e0"] * len(row)
#                         return [""] * len(row)

#                     st.dataframe(
#                         df.style.apply(highlight_cracked, axis=1),
#                         use_container_width=True,
#                     )

#     except Exception as e:
#         st.error(f"Error processing the uploaded image: {e}")
#         aggressive_cleanup()
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
# import io
# import math
# import cv2
# import matplotlib.cm as cm
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from keras.models import Model
# from PIL import Image, ImageOps, ExifTags
# from streamlit_image_comparison import image_comparison
# import pandas as pd
# import gc
# import hashlib

# TILE_SIZE = 224  # Each tile is 224x224 pixels


# # ══════════════════════════════════════════════
# # Model loading  (cached for the whole session)
# # ══════════════════════════════════════════════

# @st.cache_resource
# def load_model():
#     try:
#         model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None


# @st.cache_resource
# def build_custom_model(sensitivity: int):
#     """Return a two-output Model cached by sensitivity level."""
#     model = load_model()
#     return Model(
#         inputs=model.inputs,
#         outputs=(model.layers[sensitivity].output, model.layers[-1].output),
#     )


# model = load_model()


# # ══════════════════════════════════════════════
# # Helper utilities
# # ══════════════════════════════════════════════

# def correct_orientation(image):
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = image._getexif()
#         if exif is not None:
#             orientation_val = exif.get(orientation, 1)
#             if orientation_val == 3:
#                 image = image.rotate(180, expand=True)
#             elif orientation_val == 6:
#                 image = image.rotate(270, expand=True)
#             elif orientation_val == 8:
#                 image = image.rotate(90, expand=True)
#     except (AttributeError, KeyError, IndexError):
#         pass
#     return image


# def add_canvas(image, fill_color=(255, 255, 255)):
#     image_width, image_height = image.size
#     canvas_width  = image_width  + math.ceil(0.015 * image_width)
#     canvas_height = image_height + math.ceil(0.07  * image_height)
#     canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
#     paste_position = (
#         (canvas_width  - image_width)  // 2,
#         (canvas_height - image_height) // 7,
#     )
#     canvas.paste(image, paste_position)
#     return canvas


# def add_white_border(image, border_size):
#     return ImageOps.expand(image, border=border_size, fill=(255, 255, 255))


# # ══════════════════════════════════════════════
# # Whole-image prediction
# # ══════════════════════════════════════════════

# @st.cache_data(show_spinner=False)
# def import_and_predict(image_bytes: bytes, sensitivity: int = 9):
#     """
#     Accepts raw image bytes so that st.cache_data can hash the input reliably.
#     Returns (pred_vec, image_with_border, contours_with_border,
#              heatmap_image, contoured_image, overlay_img)
#     """
#     try:
#         image_data   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         original_img = np.array(image_data)

#         orig_height, orig_width, _ = original_img.shape
#         max_dimension     = max(orig_width, orig_height)
#         contour_thickness = max(2, int(max_dimension / 200))

#         img_resized  = cv2.resize(original_img, (224, 224))
#         img_tensor   = np.expand_dims(img_resized, axis=0).astype(np.float16) / 255.0

#         custom_model              = build_custom_model(sensitivity)
#         conv2d_3_output, pred_vec = custom_model.predict(img_tensor, verbose=0)
#         conv2d_3_output           = np.squeeze(conv2d_3_output)

#         heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height),
#                                       interpolation=cv2.INTER_LINEAR)
#         heat_map = np.mean(heat_map_resized, axis=-1)
#         heat_map = np.maximum(heat_map, 0)
#         heat_map = heat_map / heat_map.max()

#         heat_map_thresh = np.uint8(255 * heat_map)
#         _, thresh_map   = cv2.threshold(heat_map_thresh, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#         contours, _     = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
#         heatmap_image   = Image.fromarray(heatmap_colored)

#         contoured_img = original_img.copy()
#         cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)
#         contoured_image = Image.fromarray(contoured_img)

#         heatmap_image_rgba  = heatmap_image.convert("RGBA")
#         original_img_pil    = Image.fromarray(original_img).convert("RGBA")
#         heatmap_overlay     = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

#         heatmap_overlay_rgb    = heatmap_overlay.convert("RGB")
#         heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
#         cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)
#         overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

#         border_size          = 10
#         image_with_border    = add_white_border(image_data, border_size)
#         contours_with_border = add_white_border(overlay_img, border_size)

#         return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")
#         return None, None, None, None, None, None


# # ══════════════════════════════════════════════
# # Tile-based helpers
# # ══════════════════════════════════════════════

# def predict_tiles_batch(tiles_np, sensitivity=9):
#     batch        = np.stack(tiles_np, axis=0) / 255.0
#     custom_model = build_custom_model(sensitivity)
#     conv_outputs, pred_vecs = custom_model.predict(batch, verbose=0)
#     pred_indices = [int(np.argmax(pv)) for pv in pred_vecs]
#     return pred_indices, pred_vecs, conv_outputs


# def ensemble_predict_tiles(tiles_np, sensitivity_levels=(7, 9, 11)):
#     """Soft-voting ensemble. Batch tensor is built once outside the loop."""
#     batch               = np.stack(tiles_np, axis=0) / 255.0   # built ONCE
#     mid_idx             = len(sensitivity_levels) // 2
#     all_pred_vecs       = []
#     middle_conv_outputs = None

#     for i, sens in enumerate(sensitivity_levels):
#         custom_model             = build_custom_model(sens)     # reuses cached model
#         conv_outputs, pred_vecs  = custom_model.predict(batch, verbose=0)
#         all_pred_vecs.append(pred_vecs)
#         if i == mid_idx:
#             middle_conv_outputs = conv_outputs

#     averaged_pred_vecs = np.mean(all_pred_vecs, axis=0)
#     pred_indices       = [int(np.argmax(pv)) for pv in averaged_pred_vecs]
#     return pred_indices, averaged_pred_vecs, middle_conv_outputs


# # ══════════════════════════════════════════════
# # Tile-based crack detection  (main function)
# # ══════════════════════════════════════════════

# @st.cache_data(show_spinner=False)
# def tiled_crack_detection(image_bytes: bytes,
#                            sensitivity: int = 9,
#                            confidence_threshold: float = 95.0,
#                            use_ensemble: bool = False,
#                            ensemble_levels: tuple = (7, 9, 11)):
#     """
#     All parameters are hashable so st.cache_data works correctly.
#     Returns (result_image, tile_grid_image, contours_only_image, numbered_image, summary).
#     """
#     image_data   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     original_img = np.array(image_data)
#     orig_h, orig_w, _ = original_img.shape

#     pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
#     pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE
#     padded_img = cv2.copyMakeBorder(original_img, 0, pad_h, 0, pad_w,
#                                     cv2.BORDER_REFLECT)
#     pad_h_total, pad_w_total = padded_img.shape[:2]

#     n_rows      = pad_h_total // TILE_SIZE
#     n_cols      = pad_w_total // TILE_SIZE
#     total_tiles = n_rows * n_cols

#     contour_thickness = max(2, int(max(orig_w, orig_h) / 200))

#     output_canvas     = padded_img.copy()
#     tile_grid_overlay = padded_img.copy().astype(np.float16)

#     CLASS_COLORS_BGR = {
#         0: (0,   200,  0),
#         1: (0,   0,   255),
#         2: (0,   165, 255),
#     }
#     CLASS_LABELS = {0: "Normal", 1: "Cracked", 2: "Not a Wall"}

#     tile_results    = []
#     cracked_count   = 0
#     MINI_BATCH_SIZE = 8

#     tile_coords = []
#     tiles_np    = []
#     for r in range(n_rows):
#         for c in range(n_cols):
#             y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
#             x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
#             tile_coords.append((r, c, y0, y1, x0, x1))
#             tiles_np.append(padded_img[y0:y1, x0:x1].astype(np.float16))

#     all_pred_indices = []
#     all_pred_vecs    = []
#     all_conv_outputs = []

#     for batch_start in range(0, total_tiles, MINI_BATCH_SIZE):
#         batch_end   = min(batch_start + MINI_BATCH_SIZE, total_tiles)
#         batch_tiles = tiles_np[batch_start:batch_end]

#         if use_ensemble:
#             b_pred_indices, b_pred_vecs, b_conv_outputs = ensemble_predict_tiles(
#                 batch_tiles, sensitivity_levels=ensemble_levels
#             )
#         else:
#             b_pred_indices, b_pred_vecs, b_conv_outputs = predict_tiles_batch(
#                 batch_tiles, sensitivity
#             )

#         all_pred_indices.extend(b_pred_indices)
#         all_pred_vecs.append(b_pred_vecs)
#         all_conv_outputs.append(b_conv_outputs)

#     pred_indices = all_pred_indices
#     pred_vecs    = np.concatenate(all_pred_vecs,    axis=0)
#     conv_outputs = np.concatenate(all_conv_outputs, axis=0)

#     for tile_idx, (r, c, y0, y1, x0, x1) in enumerate(tile_coords):
#         pred_index  = pred_indices[tile_idx]
#         pred_vec    = pred_vecs[tile_idx]
#         conv_output = conv_outputs[tile_idx]

#         conf = float(pred_vec[pred_index]) * 100
#         if pred_index == 1 and conf < confidence_threshold:
#             pred_index = 0

#         tile_results.append({
#             "row":        r,
#             "col":        c,
#             "pred":       pred_index,
#             "label":      CLASS_LABELS[pred_index],
#             "confidence": conf,
#         })

#         color_bgr = CLASS_COLORS_BGR[pred_index]
#         alpha     = 0.35
#         tile_grid_overlay[y0:y1, x0:x1] = (
#             (1 - alpha) * tile_grid_overlay[y0:y1, x0:x1].astype(np.float32)
#             + alpha * np.array(color_bgr[::-1], dtype=np.float32)
#         )

#         cv2.rectangle(output_canvas, (x0, y0), (x1 - 1, y1 - 1), color_bgr[::-1], 2)
#         cv2.rectangle(tile_grid_overlay.astype(np.uint8),
#                       (x0, y0), (x1 - 1, y1 - 1), color_bgr[::-1], 2)

#         if pred_index == 1:
#             cracked_count += 1
#             heat = np.mean(conv_output, axis=-1) if conv_output.ndim == 3 else conv_output
#             heat = np.maximum(heat, 0)
#             if heat.max() > 0:
#                 heat = heat / heat.max()
#             heat_resized  = cv2.resize(heat, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
#             heat_uint8    = np.uint8(255 * heat_resized)
#             _, thresh_map = cv2.threshold(heat_uint8, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#             contours, _   = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             shifted = [cnt + np.array([[[x0, y0]]]) for cnt in contours]
#             cv2.drawContours(output_canvas, shifted, -1, (255, 0, 0), contour_thickness)

#     # ── Contours-only image ───────────────────────────────────────────────
#     contours_only_canvas = padded_img.copy()
#     for tile_idx2, (r2, c2, y0t, y1t, x0t, x1t) in enumerate(tile_coords):
#         t = tile_results[tile_idx2]
#         if t["pred"] == 1:
#             conv_out2 = conv_outputs[tile_idx2]
#             heat2     = np.mean(conv_out2, axis=-1) if conv_out2.ndim == 3 else conv_out2
#             heat2     = np.maximum(heat2, 0)
#             if heat2.max() > 0:
#                 heat2 = heat2 / heat2.max()
#             heat2_resized = cv2.resize(heat2, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
#             heat2_uint8   = np.uint8(255 * heat2_resized)
#             _, thresh2    = cv2.threshold(heat2_uint8, int(255 * 0.5), 255, cv2.THRESH_BINARY)
#             contours2, _  = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             shifted2 = [cnt + np.array([[[x0t, y0t]]]) for cnt in contours2]
#             cv2.drawContours(contours_only_canvas, shifted2, -1, (255, 0, 0), contour_thickness)
#     contours_only_image = Image.fromarray(contours_only_canvas[:orig_h, :orig_w])

#     # ── Numbered tile grid ────────────────────────────────────────────────
#     numbered_canvas = padded_img.copy()
#     font            = cv2.FONT_HERSHEY_SIMPLEX
#     tile_number     = 0
#     max_img_dim     = max(orig_w, orig_h)
#     font_scale      = max(0.4, max_img_dim / 1500)
#     font_thickness  = max(1, int(font_scale * 2.5))
#     for r2 in range(n_rows):
#         for c2 in range(n_cols):
#             y0t    = r2 * TILE_SIZE
#             x0t    = c2 * TILE_SIZE
#             t_info = next(t for t in tile_results if t["row"] == r2 and t["col"] == c2)
#             color_bgr = CLASS_COLORS_BGR[t_info["pred"]]
#             color_rgb = color_bgr[::-1]
#             cv2.rectangle(numbered_canvas,
#                           (x0t, y0t), (x0t + TILE_SIZE - 1, y0t + TILE_SIZE - 1),
#                           color_rgb, 2)
#             label_str   = str(tile_number)
#             padding     = max(4, int(TILE_SIZE * 0.04))
#             (tw, th), _ = cv2.getTextSize(label_str, font, font_scale, font_thickness)
#             tx = x0t + padding
#             ty = y0t + th + padding
#             cv2.putText(numbered_canvas, label_str, (tx + 1, ty + 1),
#                         font, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
#             cv2.putText(numbered_canvas, label_str, (tx, ty),
#                         font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#             tile_number += 1
#     numbered_image = Image.fromarray(numbered_canvas[:orig_h, :orig_w])

#     result_image    = Image.fromarray(output_canvas[:orig_h, :orig_w])
#     tile_grid_image = Image.fromarray(tile_grid_overlay.astype(np.uint8)[:orig_h, :orig_w])

#     summary = {
#         "total":    total_tiles,
#         "cracked":  cracked_count,
#         "normal":   sum(1 for t in tile_results if t["pred"] == 0),
#         "not_wall": sum(1 for t in tile_results if t["pred"] == 2),
#         "tiles":    tile_results,
#         "grid":     (n_rows, n_cols),
#     }
#     tf.keras.backend.clear_session()
#     gc.collect()                           
#     return result_image, tile_grid_image, contours_only_image, numbered_image, summary


# # ══════════════════════════════════════════════
# # Session-state initialisation
# # ══════════════════════════════════════════════

# def init_session_state():
#     defaults = {
#         "whole_image_results":  None,   # cached whole-image output tuple
#         "tile_results":         None,   # cached tile output tuple
#         "last_image_hash":      None,   # detect new file upload
#         # Settings snapshot at the time of the LAST run — for stale detection
#         "run_sensitivity":      None,
#         "run_confidence":       None,
#         "run_ensemble":         None,
#         "run_ensemble_lvls":    None,
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# init_session_state()


# # ══════════════════════════════════════════════
# # Main Streamlit UI
# # ══════════════════════════════════════════════

# file = st.file_uploader(
#     "Please upload an image of the brick wall",
#     type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"),
# )

# if file is None:
#     st.info("Please upload an image file to start the detection.")
# else:
#     image_bytes = file.getvalue()
#     image_hash = hashlib.md5(image_bytes).hexdigest()

#     # ── Reset everything when a new image is uploaded ──────────────────
#     if st.session_state["last_image_hash"] != image_hash:
#         st.session_state["whole_image_results"] = None
#         st.session_state["tile_results"]        = None
#         st.session_state["run_sensitivity"]     = None
#         st.session_state["run_confidence"]      = None
#         st.session_state["run_ensemble"]        = None
#         st.session_state["run_ensemble_lvls"]   = None
#         st.session_state["last_image_hash"]     = image_hash

#     try:
#         image = Image.open(io.BytesIO(image_bytes))
#         if image is None:
#             raise ValueError("Uploaded file is not a valid image.")
#         image = correct_orientation(image)

#         # ── Auto-resize large images ───────────────────────────────────
#         MAX_DIMENSION = 2000
#         orig_w, orig_h = image.size
#         if max(orig_w, orig_h) > MAX_DIMENSION:
#             scale = MAX_DIMENSION / max(orig_w, orig_h)
#             new_w = int(orig_w * scale)
#             new_h = int(orig_h * scale)
#             image = image.resize((new_w, new_h), Image.LANCZOS)
#             st.info(
#                 f"📐 Image resized from {orig_w}×{orig_h} to {new_w}×{new_h} px "
#                 f"to fit within memory limits."
#             )
#             buf = io.BytesIO()
#             image.save(buf, format="PNG")
#             image_bytes = buf.getvalue()

#         # ══════════════════════════════════════════════════════════════
#         # Settings expander — only reads values, NEVER triggers inference
#         # ══════════════════════════════════════════════════════════════
#         with st.expander("🔍 Sensitivity Settings"):
#             sensitivity = st.slider(
#                 "Adjust Detection Sensitivity (Higher = more sensitive)",
#                 min_value=0, max_value=12, value=9, step=1,
#             )
#             st.divider()
#             confidence_threshold = st.slider(
#                 "🎚️ Crack Confidence Threshold (%)",
#                 min_value=10.0, max_value=99.0, value=95.0, step=1.0,
#                 help=(
#                     "Tiles predicted as 'Cracked' below this confidence are "
#                     "reclassified as Normal. Lower = more tiles flagged."
#                 ),
#             )
#             st.divider()
#             use_ensemble = st.toggle(
#                 "🧪 Ensemble Mode (average predictions across multiple sensitivity levels)",
#                 value=False,
#             )
#             if use_ensemble:
#                 ensemble_levels = tuple(sorted(st.multiselect(
#                     "Sensitivity levels to ensemble",
#                     options=list(range(0, 13)),
#                     default=[7, 9, 11],
#                     help="Select at least 2 levels.",
#                 )))
#                 if len(ensemble_levels) < 2:
#                     st.warning("⚠️ Please select at least 2 sensitivity levels.")
#                     use_ensemble   = False
#                     ensemble_levels = ()    # empty tuple — ensemble effectively off
#             else:
#                 # FIX: keep empty; do NOT set to [sensitivity].
#                 # Prevents sensitivity slider changes from falsely triggering
#                 # stale warnings when ensemble mode is off.
#                 ensemble_levels = ()

#         # ══════════════════════════════════════════════════════════════
#         # Whole-image analysis — ONLY runs on explicit button click
#         # ══════════════════════════════════════════════════════════════
#         run_whole = st.button("🔬 Run / Refresh Whole-Image Analysis", type="secondary")

#         # FIX: removed automatic `or results is None` trigger.
#         # Nothing runs until the user explicitly clicks the button.
#         if run_whole:
#             with st.spinner("Running whole-image analysis…"):
#                 results = import_and_predict(image_bytes, sensitivity)
#             st.session_state["whole_image_results"] = results
#             st.session_state["run_sensitivity"]     = sensitivity
#             st.session_state["run_ensemble"]        = use_ensemble
#             st.session_state["run_ensemble_lvls"]   = ensemble_levels

#         # ── Stale warning for whole-image results ──────────────────────
#         if st.session_state["whole_image_results"] is not None:
#             whole_stale = (
#                 st.session_state["run_sensitivity"] != sensitivity
#                 or st.session_state["run_ensemble"] != use_ensemble
#                 # FIX: only compare ensemble_levels when ensemble is actually ON
#                 or (use_ensemble and st.session_state["run_ensemble_lvls"] != ensemble_levels)
#             )
#             if whole_stale:
#                 st.warning(
#                     "⚠️ Settings have changed since the last whole-image analysis. "
#                     "Click **🔬 Run / Refresh Whole-Image Analysis** to update."
#                 )

#         # ── Prompt if no results yet ───────────────────────────────────
#         if st.session_state["whole_image_results"] is None:
#             st.info("Click **🔬 Run / Refresh Whole-Image Analysis** to analyse the uploaded image.")
#         else:
#             (predictions, image_with_border, contours_with_border,
#              heatmap_image, contoured_image, overlay_img) = st.session_state["whole_image_results"]

#             if predictions is not None:
#                 predicted_class        = np.argmax(predictions)
#                 prediction_percentages = predictions[0] * 100

#                 if predicted_class == 0:
#                     st.success("✅ This is a normal brick wall.")
#                 elif predicted_class == 1:
#                     st.error("❌ This wall is a cracked brick wall.")
#                 elif predicted_class == 2:
#                     st.warning("⚠️ This is not a brick wall.")
#                 else:
#                     st.error(f"❓ Unknown prediction result: {predicted_class}")

#                 st.write("**Prediction Percentages:**")
#                 st.markdown(f"""
#                     <div style="display:flex;justify-content:space-between;font-size:14px;
#                                 color:#e0e0e0;background-color:#808080;padding:3px;border-radius:9px;">
#                         <div style="text-align:center;flex:1;">
#                             🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%
#                         </div>
#                         <div style="text-align:center;flex:1;">
#                             🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%
#                         </div>
#                         <div style="text-align:center;flex:1;">
#                             🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 st.write("")
#                 st.subheader("🔎 Whole-Image Analysis")
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.image(image, caption="Uploaded Image", use_container_width=True)
#                 with col2:
#                     if predicted_class == 1:
#                         st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)
#                 with col3:
#                     if predicted_class == 1:
#                         st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)
#                 with col4:
#                     if predicted_class == 1:
#                         st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
#                     else:
#                         st.image(image,
#                                  caption="No cracks detected" if predicted_class == 0 else "No wall detected",
#                                  use_container_width=True)

#                 # ── Before / after slider ──────────────────────────────
#                 image_with_border    = add_canvas(image_with_border)
#                 contours_with_border = add_canvas(contours_with_border)
#                 st.write("")
#                 if st.checkbox("Original vs Cracked Slider"):
#                     st.markdown(
#                         "<style>.centered-image-container{display:flex;justify-content:center;}"
#                         "</style><div class='centered-image-container'>",
#                         unsafe_allow_html=True,
#                     )
#                     if predicted_class == 1:
#                         image_comparison(
#                             img1=image_with_border, img2=contours_with_border,
#                             label1="Uploaded Image", label2="Cracks Localization",
#                             show_labels=False,
#                         )
#                     else:
#                         image_comparison(
#                             img1=image_with_border, img2=image_with_border,
#                             label1="Uploaded Image", label2="Cracks Localization",
#                             show_labels=False,
#                         )
#                     st.markdown("</div>", unsafe_allow_html=True)

#         # ══════════════════════════════════════════════════════════════
#         # Tile-based section
#         # ══════════════════════════════════════════════════════════════
#         st.divider()
#         st.subheader("🧩 Tile-Based Segment Analysis (224 × 224 px tiles)")
#         st.caption(
#             "The image is divided into 224 × 224 pixel tiles. "
#             "Each tile is independently classified. "
#             "🟢 Green = Normal  |  🔴 Red = Cracked  |  🟠 Orange = Not a wall"
#         )

#         run_tiled = st.button("▶ Run Tile-Based Analysis", type="primary")

#         if run_tiled:
#             if use_ensemble:
#                 st.info(
#                     f"🧪 Ensemble mode active — averaging predictions across "
#                     f"sensitivity levels: {sorted(ensemble_levels)}"
#                 )
#             with st.spinner("Running tile analysis … this may take a moment."):
#                 tile_output = tiled_crack_detection(
#                     image_bytes,
#                     sensitivity=sensitivity,
#                     confidence_threshold=confidence_threshold,
#                     use_ensemble=use_ensemble,
#                     ensemble_levels=ensemble_levels,
#                 )
#                 gc.collect()
#             st.session_state["tile_results"]      = tile_output
#             st.session_state["run_sensitivity"]   = sensitivity
#             st.session_state["run_confidence"]    = confidence_threshold
#             st.session_state["run_ensemble"]      = use_ensemble
#             st.session_state["run_ensemble_lvls"] = ensemble_levels

#         # ── Stale warning for tile results ─────────────────────────────
#         if st.session_state["tile_results"] is not None:
#             tile_stale = (
#                 st.session_state["run_sensitivity"] != sensitivity
#                 or st.session_state["run_confidence"] != confidence_threshold
#                 or st.session_state["run_ensemble"]   != use_ensemble
#                 # FIX: only compare ensemble_levels when ensemble is actually ON
#                 or (use_ensemble and st.session_state["run_ensemble_lvls"] != ensemble_levels)
#             )
#             if tile_stale:
#                 st.warning(
#                     "⚠️ Settings have changed since the last tile analysis. "
#                     "Click **▶ Run Tile-Based Analysis** to update the results."
#                 )

#         # ── Prompt if no tile results yet ──────────────────────────────
#         if st.session_state["tile_results"] is None:
#             st.info("Click **▶ Run Tile-Based Analysis** to run tile-based crack detection.")
#         else:
#             tiled_result, tile_grid_img, contours_only_img, numbered_img, summary = \
#                 st.session_state["tile_results"]

#             m1, m2, m3, m4 = st.columns(4)
#             m1.metric("Total Tiles",   summary["total"])
#             m2.metric("🔴 Cracked",    summary["cracked"],
#                       delta=f"{summary['cracked'] / summary['total'] * 100:.1f}%",
#                       delta_color="inverse")
#             m3.metric("🟢 Normal",     summary["normal"])
#             m4.metric("🟠 Not a Wall", summary["not_wall"])

#             cracked_pct = summary["cracked"] / summary["total"] * 100
#             if summary["cracked"] == 0:
#                 st.success("✅ No cracked tiles detected across the entire image.")
#             elif cracked_pct < 25:
#                 st.warning(f"⚠️ Minor cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
#             elif cracked_pct < 60:
#                 st.error(f"❌ Moderate cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
#             else:
#                 st.error(f"🚨 Severe cracking detected: {cracked_pct:.1f}% of tiles are cracked.")

#             st.write("")
#             tc1, tc2 = st.columns(2)
#             with tc1:
#                 st.image(tile_grid_img,
#                          caption="Tile Grid — colour-coded by class",
#                          use_container_width=True)
#             with tc2:
#                 st.image(tiled_result,
#                          caption="Contour lines drawn in cracked tiles",
#                          use_container_width=True)

#             st.write("")
#             tc3, tc4 = st.columns(2)
#             with tc3:
#                 st.image(contours_only_img,
#                          caption="Crack contours on original image (no tinting)",
#                          use_container_width=True)
#             with tc4:
#                 st.image(numbered_img,
#                          caption="Numbered tiles (colour = predicted class)",
#                          use_container_width=True)

#             with st.expander("📋 Tile-by-Tile Results"):
#                 df = pd.DataFrame(summary["tiles"])
#                 df.columns = ["Row", "Col", "Pred Index", "Label", "Confidence (%)"]
#                 df["Confidence (%)"] = df["Confidence (%)"].round(2)

#                 def highlight_cracked(row):
#                     if row["Label"] == "Cracked":
#                         return ["background-color: #ffe0e0"] * len(row)
#                     return [""] * len(row)

#                 st.dataframe(
#                     df.style.apply(highlight_cracked, axis=1),
#                     use_container_width=True,
#                 )

#     except Exception as e:
#         st.error(f"Error processing the uploaded image: {e}")
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________________________
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
# Tile-based crack detection helpers
# ──────────────────────────────────────────────

def build_custom_model(sensitivity=9):
    """Build and return the intermediate-output model (cached by sensitivity)."""
    return Model(
        inputs=model.inputs,
        outputs=(model.layers[sensitivity].output, model.layers[-1].output),
    )

def predict_tiles_batch(tiles_np, sensitivity=9):
    """
    Run the model on a batch of 224×224 tiles in a single forward pass.
    tiles_np : list of np.ndarray, each (224, 224, 3)
    Returns:
        pred_indices : list[int]
        pred_vecs    : np.ndarray (N, num_classes)
        conv_outputs : np.ndarray (N, H, W, C)
    """
    batch = np.stack(tiles_np, axis=0) / 255.0          # (N, 224, 224, 3)
    custom_model = build_custom_model(sensitivity)
    conv_outputs, pred_vecs = custom_model.predict(batch, verbose=0)
    pred_indices = [int(np.argmax(pv)) for pv in pred_vecs]
    return pred_indices, pred_vecs, conv_outputs


def ensemble_predict_tiles(tiles_np, sensitivity_levels=(7, 9, 11)):
    """
    Run the model at multiple sensitivity levels and average the prediction
    vectors (soft voting). Returns blended pred_indices, pred_vecs, and
    conv_outputs from the middle sensitivity level.

    tiles_np          : list of np.ndarray, each (224, 224, 3)
    sensitivity_levels: tuple/list of int layer indices to ensemble over
    Returns:
        pred_indices        : list[int]  — argmax of averaged softmax
        averaged_pred_vecs  : np.ndarray (N, num_classes)
        middle_conv_outputs : np.ndarray (N, H, W, C)  from middle level
    """
    all_pred_vecs       = []
    middle_conv_outputs = None
    mid_idx             = len(sensitivity_levels) // 2   # middle level index

    for i, sens in enumerate(sensitivity_levels):
        batch        = np.stack(tiles_np, axis=0) / 255.0
        custom_model = build_custom_model(sens)
        conv_outputs, pred_vecs = custom_model.predict(batch, verbose=0)
        all_pred_vecs.append(pred_vecs)
        if i == mid_idx:
            middle_conv_outputs = conv_outputs   # keep conv map from middle level

    # Soft voting: average softmax probabilities across all levels
    averaged_pred_vecs = np.mean(all_pred_vecs, axis=0)          # (N, num_classes)
    pred_indices       = [int(np.argmax(pv)) for pv in averaged_pred_vecs]

    return pred_indices, averaged_pred_vecs, middle_conv_outputs


# ──────────────────────────────────────────────
# Tile-based crack detection (main function)
# ──────────────────────────────────────────────

def tiled_crack_detection(image_data, sensitivity=9, progress_bar=None,
                           confidence_threshold=95.0,
                           use_ensemble=False, ensemble_levels=(7, 9, 11)):
    """
    1. Pad image so it tiles perfectly into 224×224 blocks.
    2. Run the model on each tile (single sensitivity or ensemble).
    3. For tiles predicted as 'Cracked', generate a heatmap and contours.
    4. Assemble a full-resolution output image with contours drawn only in
       cracked segments, plus a coloured tile-grid overlay.
    Returns (result_image, tile_grid_image, contours_only_image, numbered_image, summary).
    """
    original_img = np.array(image_data)
    if original_img.shape[-1] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w, _ = original_img.shape

    # ── 1. Pad to a multiple of TILE_SIZE ──────────────────────────────────
    pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE
    padded_img = cv2.copyMakeBorder(original_img, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_REFLECT)
    pad_h_total, pad_w_total = padded_img.shape[:2]

    n_rows = pad_h_total // TILE_SIZE
    n_cols = pad_w_total // TILE_SIZE
    total_tiles = n_rows * n_cols

    contour_thickness = max(2, int(max(orig_w, orig_h) / 200))

    # Output canvas: copy of padded image; we draw contours onto it
    output_canvas     = padded_img.copy()
    # Tile-grid overlay: colour each tile by class
    tile_grid_overlay = padded_img.copy().astype(np.float32)

    # Colour codes per class (BGR for OpenCV)
    CLASS_COLORS_BGR = {
        0: (0,   200,  0),    # Normal    → green
        1: (0,   0,   255),   # Cracked   → red
        2: (0,   165, 255),   # Not a wall → orange
    }
    CLASS_LABELS = {0: "Normal", 1: "Cracked", 2: "Not a Wall"}

    tile_results    = []   # list of dicts: {row, col, pred, label, confidence}
    cracked_count   = 0
    MINI_BATCH_SIZE = 64   # process this many tiles at once to limit memory

    # ── Collect all tile coordinates and pixel data ───────────────────────
    tile_coords = []   # (r, c, y0, y1, x0, x1)
    tiles_np    = []   # raw pixel arrays

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            tile_coords.append((r, c, y0, y1, x0, x1))
            tiles_np.append(padded_img[y0:y1, x0:x1])

    # ── Mini-batch forward passes ─────────────────────────────────────────
    all_pred_indices = []
    all_pred_vecs    = []
    all_conv_outputs = []

    for batch_start in range(0, total_tiles, MINI_BATCH_SIZE):
        batch_end   = min(batch_start + MINI_BATCH_SIZE, total_tiles)
        batch_tiles = tiles_np[batch_start:batch_end]

        if progress_bar is not None:
            progress_bar.progress(
                batch_start / total_tiles,
                text=f"Predicting tiles {batch_start + 1}–{batch_end} of {total_tiles} …"
            )

        # ── Choose single or ensemble prediction ──────────────────────────
        if use_ensemble:
            b_pred_indices, b_pred_vecs, b_conv_outputs = ensemble_predict_tiles(
                batch_tiles, sensitivity_levels=tuple(ensemble_levels)
            )
        else:
            b_pred_indices, b_pred_vecs, b_conv_outputs = predict_tiles_batch(
                batch_tiles, sensitivity
            )

        all_pred_indices.extend(b_pred_indices)
        all_pred_vecs.append(b_pred_vecs)
        all_conv_outputs.append(b_conv_outputs)

    # Concatenate results from all mini-batches
    pred_indices = all_pred_indices
    pred_vecs    = np.concatenate(all_pred_vecs,    axis=0)
    conv_outputs = np.concatenate(all_conv_outputs, axis=0)

    # ── Post-process each tile result ─────────────────────────────────────
    for tile_idx, (r, c, y0, y1, x0, x1) in enumerate(tile_coords):
        pred_index  = pred_indices[tile_idx]
        pred_vec    = pred_vecs[tile_idx]
        conv_output = conv_outputs[tile_idx]   # (H, W, C)

        conf = float(pred_vec[pred_index]) * 100

        # Downgrade low-confidence crack predictions to Normal
        if pred_index == 1 and conf < confidence_threshold:
            pred_index = 0

        tile_results.append({
            "row":        r,
            "col":        c,
            "pred":       pred_index,
            "label":      CLASS_LABELS[pred_index],
            "confidence": conf,
        })

        color_bgr = CLASS_COLORS_BGR[pred_index]

        # ── Colour the tile in the grid overlay ──────────────────────
        alpha = 0.35
        tile_grid_overlay[y0:y1, x0:x1] = (
            (1 - alpha) * tile_grid_overlay[y0:y1, x0:x1].astype(np.float32)
            + alpha * np.array(color_bgr[::-1], dtype=np.float32)   # BGR→RGB
        )

        # Draw tile border
        cv2.rectangle(output_canvas, (x0, y0), (x1 - 1, y1 - 1),
                      color_bgr[::-1], 2)
        cv2.rectangle(tile_grid_overlay.astype(np.uint8), (x0, y0),
                      (x1 - 1, y1 - 1), color_bgr[::-1], 2)

        # ── If cracked: generate localised contours inside this tile ─
        if pred_index == 1:
            cracked_count += 1

            heat = np.mean(conv_output, axis=-1) if conv_output.ndim == 3 else conv_output
            heat = np.maximum(heat, 0)
            if heat.max() > 0:
                heat = heat / heat.max()

            heat_resized  = cv2.resize(heat, (TILE_SIZE, TILE_SIZE),
                                       interpolation=cv2.INTER_LINEAR)
            heat_uint8    = np.uint8(255 * heat_resized)
            _, thresh_map = cv2.threshold(heat_uint8, int(255 * 0.5),
                                          255, cv2.THRESH_BINARY)
            contours, _   = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

            shifted = [cnt + np.array([[[x0, y0]]]) for cnt in contours]
            cv2.drawContours(output_canvas, shifted, -1,
                             (255, 0, 0), contour_thickness)

        # Update progress bar (post-processing phase)
        if progress_bar is not None:
            progress_bar.progress(
                (tile_idx + 1) / total_tiles,
                text=f"Post-processing tile {tile_idx + 1}/{total_tiles} …"
            )

    # ── Image 3: contours only on clean original ──────────────────────────
    contours_only_canvas = padded_img.copy()
    for tile_idx2, (r2, c2, y0t, y1t, x0t, x1t) in enumerate(tile_coords):
        t = tile_results[tile_idx2]
        if t["pred"] == 1:
            conv_out2 = conv_outputs[tile_idx2]
            heat2     = np.mean(conv_out2, axis=-1) if conv_out2.ndim == 3 else conv_out2
            heat2     = np.maximum(heat2, 0)
            if heat2.max() > 0:
                heat2 = heat2 / heat2.max()
            heat2_resized = cv2.resize(heat2, (TILE_SIZE, TILE_SIZE),
                                       interpolation=cv2.INTER_LINEAR)
            heat2_uint8   = np.uint8(255 * heat2_resized)
            _, thresh2    = cv2.threshold(heat2_uint8, int(255 * 0.5),
                                          255, cv2.THRESH_BINARY)
            contours2, _  = cv2.findContours(thresh2, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
            shifted2 = [cnt + np.array([[[x0t, y0t]]]) for cnt in contours2]
            cv2.drawContours(contours_only_canvas, shifted2, -1,
                             (255, 0, 0), contour_thickness)
    contours_only_image = Image.fromarray(contours_only_canvas[:orig_h, :orig_w])

    # ── Image 4: numbered tile grid ───────────────────────────────────────
    numbered_canvas = padded_img.copy()
    font            = cv2.FONT_HERSHEY_SIMPLEX
    tile_number     = 0
    max_img_dim     = max(orig_w, orig_h)
    font_scale      = max(0.4, max_img_dim / 1500)
    font_thickness  = max(1, int(font_scale * 2.5))
    for r2 in range(n_rows):
        for c2 in range(n_cols):
            y0t   = r2 * TILE_SIZE
            x0t   = c2 * TILE_SIZE
            t_info = next(t for t in tile_results if t["row"] == r2 and t["col"] == c2)
            color_bgr = CLASS_COLORS_BGR[t_info["pred"]]
            color_rgb = color_bgr[::-1]
            cv2.rectangle(numbered_canvas,
                          (x0t, y0t),
                          (x0t + TILE_SIZE - 1, y0t + TILE_SIZE - 1),
                          color_rgb, 2)
            label_str   = str(tile_number)
            padding     = max(4, int(TILE_SIZE * 0.04))
            (tw, th), _ = cv2.getTextSize(label_str, font, font_scale, font_thickness)
            tx = x0t + padding
            ty = y0t + th + padding
            cv2.putText(numbered_canvas, label_str, (tx + 1, ty + 1),
                        font, font_scale, (255, 255, 255), font_thickness + 1,
                        cv2.LINE_AA)
            cv2.putText(numbered_canvas, label_str, (tx, ty),
                        font, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)
            tile_number += 1
    numbered_image = Image.fromarray(numbered_canvas[:orig_h, :orig_w])

    # ── Crop back to original dimensions ─────────────────────────────────
    result_image    = Image.fromarray(output_canvas[:orig_h, :orig_w])
    tile_grid_image = Image.fromarray(
        tile_grid_overlay.astype(np.uint8)[:orig_h, :orig_w]
    )

    summary = {
        "total":    total_tiles,
        "cracked":  cracked_count,
        "normal":   sum(1 for t in tile_results if t["pred"] == 0),
        "not_wall": sum(1 for t in tile_results if t["pred"] == 2),
        "tiles":    tile_results,
        "grid":     (n_rows, n_cols),
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
                scale = MAX_DIMENSION / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)
                st.info(
                    f"📐 Image resized from {orig_w}×{orig_h} to {new_w}×{new_h} px "
                    f"to fit within memory limits. Detection quality is not affected."
                )

            # ── Whole-image prediction ─────────────────────────────────
            predictions, image_with_border, contours_with_border, \
                heatmap_image, contoured_image, overlay_img = import_and_predict(image)

            if predictions is not None:
                predicted_class        = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                if predicted_class == 0:
                    st.success("✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error("❌ This wall is a cracked brick wall.")
                elif predicted_class == 2:
                    st.warning("⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")

                st.write("**Prediction Percentages:**")
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 14px;
                                color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
                        <div style="text-align: center; flex: 1;">
                            🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%
                        </div>
                        <div style="text-align: center; flex: 1;">
                            🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%
                        </div>
                        <div style="text-align: center; flex: 1;">
                            🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")

                # ── Settings expander ──────────────────────────────────
                with st.expander("🔍 Sensitivity Settings"):
                    sensitivity = st.slider(
                        "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                        min_value=0, max_value=12, value=9, step=1, format="%.1f",
                    )
                    st.divider()
                    confidence_threshold = st.slider(
                        "🎚️ Crack Confidence Threshold (%)",
                        min_value=10.0, max_value=99.0, value=95.0, step=1.0,
                        help=(
                            "Tiles predicted as 'Cracked' below this confidence are reclassified as Normal. "
                            "Lower = more sensitive (more tiles flagged) | Higher = more conservative."
                        ),
                    )
                    st.divider()
                    use_ensemble = st.toggle(
                        "🧪 Ensemble Mode (average predictions across multiple sensitivity levels)",
                        value=False,
                        help=(
                            "Runs the model at multiple sensitivity levels and averages the results "
                            "for more robust tile predictions. Slower but more reliable."
                        ),
                    )
                    if use_ensemble:
                        ensemble_levels = st.multiselect(
                            "Sensitivity levels to ensemble",
                            options=list(range(0, 13)),
                            default=[7, 9, 11],
                            help="Select 2 or more levels. More levels = more robust but slower.",
                        )
                        if len(ensemble_levels) < 2:
                            st.warning("⚠️ Please select at least 2 sensitivity levels for ensemble.")
                            use_ensemble = False
                    else:
                        ensemble_levels = [sensitivity]   # fallback (unused when ensemble is off)

                # Re-run whole-image prediction with chosen sensitivity
                predictions, image_with_border, contours_with_border, \
                    heatmap_image, contoured_image, overlay_img = import_and_predict(
                        image, sensitivity=sensitivity
                    )

                # ── Whole-image results row ────────────────────────────
                st.subheader("🔎 Whole-Image Analysis")
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

                # Before/after slider
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
                # Tile-based section
                # ══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🧩 Tile-Based Segment Analysis (224 × 224 px tiles)")
                st.caption(
                    "The image is divided into 224 × 224 pixel tiles. "
                    "Each tile is independently classified. "
                    "🟢 Green = Normal  |  🔴 Red = Cracked  |  🟠 Orange = Not a wall"
                )

                run_tiled = st.button("▶ Run Tile-Based Analysis", type="primary")

                if run_tiled:
                    progress_bar = st.progress(0, text="Starting tile analysis …")

                    # Show ensemble info banner before running
                    if use_ensemble:
                        st.info(
                            f"🧪 Ensemble mode active — averaging predictions across "
                            f"sensitivity levels: {sorted(ensemble_levels)}"
                        )

                    tiled_result, tile_grid_img, contours_only_img, numbered_img, summary = tiled_crack_detection(
                        image,
                        sensitivity=sensitivity,
                        progress_bar=progress_bar,
                        confidence_threshold=confidence_threshold,
                        use_ensemble=use_ensemble,
                        ensemble_levels=ensemble_levels,
                    )
                    progress_bar.empty()

                    # ── Summary metrics ────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Tiles",   summary["total"])
                    m2.metric("🔴 Cracked",    summary["cracked"],
                              delta=f"{summary['cracked'] / summary['total'] * 100:.1f}%",
                              delta_color="inverse")
                    m3.metric("🟢 Normal",     summary["normal"])
                    m4.metric("🟠 Not a Wall", summary["not_wall"])

                    cracked_pct = summary["cracked"] / summary["total"] * 100
                    if summary["cracked"] == 0:
                        st.success("✅ No cracked tiles detected across the entire image.")
                    elif cracked_pct < 25:
                        st.warning(f"⚠️ Minor cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
                    elif cracked_pct < 60:
                        st.error(f"❌ Moderate cracking detected: {cracked_pct:.1f}% of tiles are cracked.")
                    else:
                        st.error(f"🚨 Severe cracking detected: {cracked_pct:.1f}% of tiles are cracked.")

                    # ── Row 1: colour-coded grid | contours+heatmap overlay ──
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

                    # ── Row 2: contours only | numbered tile grid ──────────
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

                    # ── Tile-by-tile breakdown table ───────────────────
                    with st.expander("📋 Tile-by-Tile Results"):
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
#____________________________________________________________________________________________________________________________________________________________________________________
# OLD WORKING CODE
#____________________________________________________________________________________________________________________________________________________________________________________
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

@st.cache_resource
#__________________________________________________________________________________________________________________________________________________________________________________
#For single model selection
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# Function to correct image orientation based on EXIF data
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
    
# Adding Canvas Background
def add_canvas(image, fill_color=(255, 255, 255)):
    """Automatically adjusts canvas size according to image size, with added padding and centers the image on the canvas."""
    # Get the original image size
    image_width, image_height = image.size
    
    # Calculate new canvas size with padding
    canvas_width = image_width + math.ceil(0.015 * image_width)
    canvas_height = image_height + math.ceil(0.07 * image_height)
    
    # Create a new image (canvas) with the calculated size
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    
    # Calculate the position to paste the image at the center of the canvas
    paste_position = (
        (canvas_width - image_width) // 2,
        (canvas_height - image_height) // 7
    )
    
    # Paste the original image onto the canvas
    canvas.paste(image, paste_position)
    
    return canvas


# Function to localize the crack and to make predictions using the TensorFlow model
def import_and_predict(image_data, sensitivity=9):
    try:
        # Convert image to numpy array
        original_img = np.array(image_data)

        # Convert image to RGB if it has an alpha channel
        if original_img.shape[-1] == 4:  # Check if the image has 4 channels
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        
        # Save original dimensions
        orig_height, orig_width, _ = original_img.shape
        
        # Calculate the maximum dimension of the original image
        max_dimension = max(orig_width, orig_height)
        
        # Set the scaling factor for contour line thickness based on the max dimension
        contour_thickness = max(2, int(max_dimension / 200))  # Adjust the divisor to control scaling

        # Preprocess the image for the model
        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
        preprocessed_img = img_tensor
        
        # Define a new model that outputs the conv2d_3 feature maps and the prediction
        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))  # `conv2d_3` and predictions

        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)  # (28, 28, 32) feature maps

        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        # Average all the filters from conv2d_3 to get a single activation map
        heat_map = np.mean(heat_map_resized, axis=-1)  # (orig_height, orig_width)

        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1

        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5  # Adjust this threshold if needed
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert heatmap to RGB for display
        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        
        # Convert heatmap to PIL format
        heatmap_image = Image.fromarray(heatmap_colored)
        
        # Create contoured image
        contoured_img = original_img.copy()  # Copy original image
        cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)  # Draw blue contours
        
        # Convert contoured image to PIL format
        contoured_image = Image.fromarray(contoured_img)

        # Overlay heatmap on original image
        heatmap_image_rgba = heatmap_image.convert("RGBA")
        original_img_pil = Image.fromarray(original_img).convert("RGBA")
        heatmap_overlay = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

        # Draw contours on the heatmap-overlayed image
        # Convert heatmap-overlayed image to RGB for contour drawing
        heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
        heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
        # heatmap_overlay_np = np.array(heatmap_overlay)
        cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)  # Draw blue contours

        # Convert overlay image to PIL format
        overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

        # Get the predicted class name
        class_labels = ["Normal", "Cracked", "Not a Wall"]
        predicted_class = class_labels[pred]

        # Add white borders
        border_size = 10  # Set the border size
        image_with_border = add_white_border(image_data, border_size)
        contours_with_border = add_white_border(overlay_img, border_size)

        return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img 
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None, None, None
        
# Adds border to the image

def add_white_border(image, border_size):
    """Add a white border to the image."""
    return ImageOps.expand(image, border= border_size, fill=(255, 255, 255))

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            # Try to open the uploaded image using PIL
            image = Image.open(file)
            if image is None:
                raise ValueError("Uploaded file is not a valid image.")
            
            # Correct the orientation if necessary
            image = correct_orientation(image)

            
            
            # Perform prediction
            predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image)
            if predictions is not None:
                predicted_class = np.argmax(predictions)
                prediction_percentages = predictions[0] * 100

                 # Display prediction result
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall. ")
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")

                st.write(f"**Prediction Percentages:**")
                # Display predictions in one line
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
                        <div style="text-align: center; flex: 1;">🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")  # Creates a blank line

                # st.write("")  # Creates a blank line

                # Create an expander for sensitivity adjustment
                with st.expander("🔍 Sensitivity Settings"):
                    # Add a slider for selecting the sensitivity dynamically
                    sensitivity = st.slider(
                        "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                        min_value=0,   # Minimum value for sensitivity
                        max_value=12,   # Maximum value for sensitivity
                        value=9,       # Default value for sensitivity
                        step=1,        # Step for incremental changes
                        format="%.1f"    # Format to display sensitivity with one decimal
                                            )


                # Perform prediction again
                predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image, sensitivity=sensitivity)

                #in one row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    if predicted_class == 1:
                        st.image(contoured_image, caption="Crack(s) Location", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
                        
                with col3:
                    if predicted_class == 1:
                        st.image(heatmap_image, caption="Crack(s) Heatmap", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
                
                with col4:
                    if predicted_class == 1:
                        st.image(overlay_img, caption="Crack(s) Localization", use_container_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_container_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_container_width=True)
         
                image_with_border = add_canvas(image_with_border)
                contours_with_border = add_canvas(contours_with_border)               
                st.write("")  # Creates a blank line
                if st.checkbox("Original vs Cracked Slider"):
                    # HTML/CSS for centering the image comparison component
                    center_style = """
                    <style>
                    .centered-image-container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    </style>
                    """
                    st.markdown(center_style, unsafe_allow_html=True)
                    
                    # Opening div tag to center the image comparison component
                    st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
                    
                    # Conditionally display image comparison
                    if predicted_class == 1:
                        image_comparison(
                            img1=image_with_border, 
                            img2=contours_with_border,
                            label1="Uploaded Image",
                            label2="Cracks Localization",
                            show_labels=False,
                        )
                    else:
                        image_comparison(
                            img1=image_with_border, 
                            img2=image_with_border,
                            label1="Uploaded Image",
                            label2="Cracks Localization",
                            show_labels=False,
                        )
                
#                     # Closing div tag
#                     st.markdown('</div>', unsafe_allow_html=True)
                            
        
#         except Exception as e:
#             st.error(f"Error processing the uploaded image: {e}")
