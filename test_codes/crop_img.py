import os
import cv2
import numpy as np
import paddle.inference as paddle_infer

# =====================================================================
# 0. CONFIGURABLE ZOOM-OUT PADDING CONSTANTS
# =====================================================================
# Increase these numbers to zoom out further; decrease them to crop tighter.
ZOOM_PAD_WIDTH = 50  # Adds extra pixel padding to the left and right ends of the line
ZOOM_PAD_HEIGHT = 35  # Adds extra pixel padding to the top and bottom (protects character parts)
MASK_DILATION = 45  # Expands the protective mask shield to prevent white-out clipping

# =====================================================================
# 1. INITIALIZE CUSTOM PADDLE DETECTION ENGINE
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "custom_tracker")

model_file = os.path.join(MODEL_DIR, "inference.json")
params_file = os.path.join(MODEL_DIR, "inference.pdiparams")

print("⏳ Loading custom Paddle Detection Engine...")
config = paddle_infer.Config(model_file, params_file)

config.disable_gpu()
config.set_cpu_math_library_num_threads(4)
config.switch_ir_optim(False)

# Evict bugged oneDNN graph passes to prevent CPU execution crashes
onednn_passes = [
    "onednn_placement_pass", "depthwise_conv_onednn_pass",
    "squeeze_transpose_onednn_fuse_pass", "conv2d_bn_onednn_fuse_pass",
    "conv2d_bias_bn_onednn_fuse_pass", "conv_elementwise_add_onednn_fuse_pass",
    "conv_activation_mkldnn_fuse_pass", "conv_concat_activation_mkldnn_fuse_pass",
    "fc_onednn_enable_pass", "elementwise_act_onednn_fuse_pass",
    "operator_scale_onednn_fuse_pass", "operator_unsqueeze_onednn_fuse_pass",
    "operator_reshape_onednn_fuse_pass"
]
for pass_name in onednn_passes:
    try:
        config.delete_pass(pass_name)
    except Exception:
        pass

paddle_predictor = paddle_infer.create_predictor(config)
print("🎯 Detection Engine Loaded Successfully!\n")


# =====================================================================
# PERSPECTIVE WARPING UTILITY FOR BENDED TEXT
# =====================================================================
def deskew_and_crop(image, pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    src_pts = np.array([tl, tr, br, bl], dtype="float32")

    # Compute the true width and height of the rotated line
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = max(int(width_top), int(width_bottom))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_h = max(int(height_right), int(height_left))

    # Destination mapping coordinates for a flat rectangle
    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_crop = cv2.warpPerspective(image, transform_matrix, (max_w, max_h))

    return warped_crop


# =====================================================================
# 2. ADVANCED LINE EXTRACTION ENGINE (ZOOM-OUT VERSION)
# =====================================================================
def run_perfect_line_cropper(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    crop_output_dir = os.path.join(BASE_DIR, "cropped_lines")
    os.makedirs(crop_output_dir, exist_ok=True)

    orig_img = cv2.imread(image_path)
    orig_h, orig_w, _ = orig_img.shape
    print(f"🖼️ Loaded Image: {os.path.basename(image_path)} ({orig_w}x{orig_h})")

    # Resize canvas matrix to match 640x640 model input bounds
    resized_img = cv2.resize(orig_img, (640, 640))
    img_data = resized_img.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    # Run inference pass
    input_names = paddle_predictor.get_input_names()
    input_handle = paddle_predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(img_data)
    paddle_predictor.run()

    output_names = paddle_predictor.get_output_names()
    output_handle = paddle_predictor.get_output_handle(output_names[0])
    paddle_output = output_handle.copy_to_cpu()

    line_crops_data = []

    # Process Segmentation Maps (DBNet Layout structures)
    if len(paddle_output.shape) == 4:
        prob_map = paddle_output[0, 0]

        # Binary thresholding setup
        _, binary_mask = cv2.threshold((prob_map * 255).astype(np.uint8), 60, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        mask_resized = cv2.resize(binary_mask, (orig_w, orig_h))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 150:
                continue

            # Extract raw minimal area layout specifications
            rect = cv2.minAreaRect(cnt)
            center, size, angle = rect

            # ADJUSTMENT FEATURE: Zoom out the box parameters safely from the center pivot point
            padded_size = (size[0] + ZOOM_PAD_WIDTH, size[1] + ZOOM_PAD_HEIGHT)
            padded_rect = (center, padded_size, angle)

            # Calculate the adjusted 4 corner points based on zoom values
            box_points = cv2.boxPoints(padded_rect)
            box_points = np.intp(box_points)

            line_crops_data.append({
                'box_points': box_points,
                'contour': cnt,
                'sort_y': center[1]
            })

    # Sort text paths chronologically top-to-bottom
    line_crops_data = sorted(line_crops_data, key=lambda item: item['sort_y'])

    # ──── EXECUTE EXTRACTION ENGINE ────
    exported_count = 0
    for item in line_crops_data:
        pts = item['box_points']
        cnt = item['contour']

        exported_count += 1

        # 1. Create isolation canvas mask
        single_line_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(single_line_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # ADJUSTMENT FEATURE: Expand mask dilation kernel to encompass zoomed boundaries
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MASK_DILATION, MASK_DILATION))
        single_line_mask = cv2.dilate(single_line_mask, dilation_kernel)

        # 2. Set fallback background to absolute clear white
        pure_white_bg = np.ones_like(orig_img) * 255

        # 3. Mask-gate: Merge layers safely
        isolated_canvas = np.where(single_line_mask[:, :, None] == 255, orig_img, pure_white_bg)

        # 4. Deskew and crop the bended canvas view window
        try:
            cropped_line = deskew_and_crop(isolated_canvas, pts)

            crop_filename = os.path.join(crop_output_dir, f"line_{exported_count}.png")
            cv2.imwrite(crop_filename, cropped_line)
            print(f"💾 Zoomed Export -> {os.path.basename(crop_filename)}")
        except Exception:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_line = isolated_canvas[
                max(0, y - 20):min(orig_h, y + h + 20), max(0, x - 20):min(orig_w, x + w + 20)]
            crop_filename = os.path.join(crop_output_dir, f"line_{exported_count}.png")
            cv2.imwrite(crop_filename, cropped_line)
            print(f"💾 Fallback Zoomed Export -> {os.path.basename(crop_filename)}")

    print(f"\n✅ Complete! Zoomed crops exported successfully to: {crop_output_dir}")


# =====================================================================
# 3. RUNTIME EXECUTION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    TARGET_IMAGE = r"C:\Users\PC\Downloads\Gemini_Generated_Image_vo88rgvo88rgvo88.png"
    run_perfect_line_cropper(TARGET_IMAGE)