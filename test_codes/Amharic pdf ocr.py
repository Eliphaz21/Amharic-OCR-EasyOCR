import os
import sys
import argparse
import yaml
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# ── local imports from your codebase ─────────────────────────────────────────
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate
from model import Model
import paddle.inference as paddle_infer

# =====================================================================
# 0. GLOBAL ZOOM-OUT PADDING CONFIGURATIONS
# =====================================================================
ZOOM_PAD_WIDTH = 50  # Extra padding at left/right line margins
ZOOM_PAD_HEIGHT = 35  # Protects tall Amharic character ascenders/descenders
MASK_DILATION = 45  # Safety pixel cushion radius for the isolation mask


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def load_opt(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    opt = AttrDict(cfg)
    if opt.get('lang_char') and opt.lang_char != 'None':
        opt.character = opt.number + opt.symbol + opt.lang_char
    elif not opt.get('character'):
        raise ValueError("Config must have 'character' or 'number'+'symbol'+'lang_char'.")
    return opt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PDF / IMAGE INPUT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_input(input_path, dpi=300, pages_filter=None, poppler_path=None):
    ext = os.path.splitext(input_path)[1].lower()
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}

    if ext == '.pdf':
        from pdf2image import convert_from_path
        print(f"[Stage 1] PDF detected — converting to images at {dpi} DPI ...")
        pages = convert_from_path(
            input_path,
            dpi=dpi,
            poppler_path=poppler_path if poppler_path else None
        )
        print(f"          {len(pages)} page(s) found.")

        if pages_filter:
            page_nums = [int(p.strip()) - 1 for p in pages_filter.split(',')]
            pages = [pages[i] for i in page_nums if 0 <= i < len(pages)]
            print(f"          Processing {len(pages)} selected page(s).")
        return pages

    elif ext in image_extensions:
        print(f"[Stage 1] Image detected — loading directly ...")
        img = Image.open(input_path).convert('RGB')
        print(f"          Image size: {img.size[0]}×{img.size[1]} px")
        return [img]
    else:
        print(f"Unsupported file type: {ext}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — INITIALIZE CUSTOM PADDLE DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_paddle_detector():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "custom_tracker")

    model_file = os.path.join(MODEL_DIR, "inference.json")
    params_file = os.path.join(MODEL_DIR, "inference.pdiparams")

    if not os.path.exists(model_file) or not os.path.exists(params_file):
        raise FileNotFoundError(f"Paddle inference files missing inside {MODEL_DIR}")

    print(f"[Stage 2] Loading custom Paddle Detection Engine from {MODEL_DIR} ...")
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

    predictor = paddle_infer.create_predictor(config)
    print("          Paddle Detection Engine ready.")
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ADVANCED PERSPECTIVE WARP ENGINE FOR CROPPING BENDED LINES
# ─────────────────────────────────────────────────────────────────────────────

def deskew_and_crop(image, pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    src_pts = np.array([tl, tr, br, bl], dtype="float32")

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = max(int(width_top), int(width_bottom))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_h = max(int(height_right), int(height_left))

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_crop = cv2.warpPerspective(image, transform_matrix, (max_w, max_h))
    return warped_crop


def extract_isolated_lines(paddle_predictor, pil_page_image):
    """
    Runs line segmentation tracking over a page image using Paddle Inference,
    isolates bended structures with adaptive dilation, crops/deskews them via OBB,
    and returns a clean list of in-memory grayscale PIL line images for recognition.
    """
    orig_img = np.array(pil_page_image.convert('RGB'))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_h, orig_w, _ = orig_img.shape

    # Preprocess image matrix for Paddle model structure (640x640, CHW)
    resized_img = cv2.resize(orig_img, (640, 640))
    img_data = resized_img.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    input_names = paddle_predictor.get_input_names()
    input_handle = paddle_predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(img_data)
    paddle_predictor.run()

    output_names = paddle_predictor.get_output_names()
    output_handle = paddle_predictor.get_output_handle(output_names[0])
    paddle_output = output_handle.copy_to_cpu()

    line_crops_data = []

    if len(paddle_output.shape) == 4:  # Segmentation Heatmap Pass
        prob_map = paddle_output[0, 0]
        _, binary_mask = cv2.threshold((prob_map * 255).astype(np.uint8), 60, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        mask_resized = cv2.resize(binary_mask, (orig_w, orig_h))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 150:
                continue

            rect = cv2.minAreaRect(cnt)
            center, size, angle = rect

            # Apply layout scale factors to expand width and height bounds
            padded_size = (size[0] + ZOOM_PAD_WIDTH, size[1] + ZOOM_PAD_HEIGHT)
            padded_rect = (center, padded_size, angle)

            box_points = cv2.boxPoints(padded_rect)
            box_points = np.intp(box_points)

            line_crops_data.append({
                'box_points': box_points,
                'contour': cnt,
                'sort_y': center[1]
            })

    # Sort lines chronologically from top to bottom
    line_crops_data = sorted(line_crops_data, key=lambda item: item['sort_y'])

    processed_pil_crops = []

    for item in line_crops_data:
        pts = item['box_points']
        cnt = item['contour']

        single_line_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(single_line_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MASK_DILATION, MASK_DILATION))
        single_line_mask = cv2.dilate(single_line_mask, dilation_kernel)

        pure_white_bg = np.ones_like(orig_img) * 255
        isolated_canvas = np.where(single_line_mask[:, :, None] == 255, orig_img, pure_white_bg)

        try:
            cropped_line = deskew_and_crop(isolated_canvas, pts)
        except Exception:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_line = isolated_canvas[
                max(0, y - 20):min(orig_h, y + h + 20), max(0, x - 20):min(orig_w, x + w + 20)]

        # Convert the cropped BGR matrix directly to an in-memory Grayscale PIL Image
        gray_line = cv2.cvtColor(cropped_line, cv2.COLOR_BGR2GRAY)
        pil_crop = Image.fromarray(gray_line)
        processed_pil_crops.append(pil_crop)

    return processed_pil_crops


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — RECOGNITION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_recognizer(model_path, opt, device):
    print(f"[Stage 4] Loading recognition model from {model_path} ...")
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    print("          Recognition model ready.")
    return model, converter


def recognize_lines(model, converter, line_images, opt, device, batch_size=32):
    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    results = []

    for start in range(0, len(line_images), batch_size):
        batch_imgs = line_images[start:start + batch_size]
        batch_tensor, _ = align([(img, '') for img in batch_imgs])
        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            if 'CTC' in opt.Prediction:
                preds = model(batch_tensor, None)
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                T = preds.size(1)
                preds_index = preds.argmax(2).data.cpu().numpy()
                preds_flat = preds_index.flatten()
                lengths = [T] * len(batch_imgs)

                preds_str = converter.decode_greedy(preds_flat, lengths)
            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length] * len(batch_imgs)).to(device)
                text_for_pred = torch.LongTensor(len(batch_imgs), opt.batch_max_length + 1).fill_(0).to(device)
                preds = model(batch_tensor, text_for_pred, is_train=False)
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                preds_str = converter.decode(preds.argmax(2), length_for_pred)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred = pred.rstrip()
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')] if '[s]' in pred else pred

            conf = pred_max_prob[:len(pred)].mean().item() if len(pred) > 0 else 0.0
            results.append((pred, conf))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — ASSEMBLE OUTPUT TEXT & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def assemble_text(page_results, min_confidence=0.3):
    pages_text = []
    for page_num, lines in enumerate(page_results, 1):
        page_lines = []
        for pred, conf in lines:
            if len(pred.strip()) == 0:
                continue
            if conf < min_confidence:
                page_lines.append(f'[? {pred} ?]')
            else:
                page_lines.append(pred)
        pages_text.append('\n'.join(page_lines))
        print(f"          Page {page_num}: {len(page_lines)} lines recognized.")
    return '\f'.join(pages_text)


def compute_metrics(page_results, ground_truth_lines=None):
    from nltk.metrics.distance import edit_distance

    all_preds = [pred for page in page_results for pred, conf in page]
    all_confs = [conf for page in page_results for pred, conf in page]
    total_lines = len(all_preds)

    if total_lines == 0:
        print("No predictions to evaluate.")
        return

    print("\n" + "=" * 65)
    print("EVALUATION METRICS")
    print("=" * 65)

    avg_conf = sum(all_confs) / total_lines
    high_conf = sum(1 for c in all_confs if c >= 0.8)
    med_conf = sum(1 for c in all_confs if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in all_confs if c < 0.5)

    print(f"\n── Confidence ───────────────────────────────────────────────")
    print(f"  Total lines recognized : {total_lines}")
    print(f"  Avg confidence         : {avg_conf:.4f}")
    print(f"  High conf (>=0.80)     : {high_conf:>5}  ({high_conf / total_lines * 100:.1f}%)")
    print(f"  Med  conf (0.50-0.80)  : {med_conf:>5}  ({med_conf / total_lines * 100:.1f}%)")
    print(f"  Low  conf (<0.50)      : {low_conf:>5}  ({low_conf / total_lines * 100:.1f}%)")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME INTERFACE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DEFAULT_INPUT = r"C:\Users\PC\Downloads\Gemini_Generated_Image_uzr7asuzr7asuzr7.png"
    DEFAULT_MODEL = r"C:\Users\PC\Downloads\best_cer (4).pth"
    DEFAULT_OUTPUT = r"output/result_croped_3.txt"
    DEFAULT_POPPLER = r"C:\Users\PC\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

    parser = argparse.ArgumentParser(description='Amharic OCR — Zoom-Isolated Pipeline')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Path to input PDF or image')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to PyTorch .pth model weights')
    parser.add_argument('--config', default='config_files/amh_filtered_config.yaml')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Path to output text file')
    parser.add_argument('--poppler_path', default=DEFAULT_POPPLER, help='Path to Poppler binaries')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for rendering PDFs')
    parser.add_argument('--batch_size', type=int, default=32, help='Recognition batch size')
    parser.add_argument('--min_conf', type=float, default=0.3, help='Min recognition confidence threshold')
    parser.add_argument('--pages', default=None, help='Target pages string list')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Input file not found: {args.input}');
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f'Model not found: {args.model}');
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f'Config not found: {args.config}');
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch Recognition Backend: {device}')

    opt = load_opt(args.config)

    # Stage 1 — Load Documents / Images
    pages = load_input(args.input, dpi=args.dpi, pages_filter=args.pages, poppler_path=args.poppler_path)

    # Stage 2 — Build Custom Paddle Detector
    paddle_predictor = build_paddle_detector()

    # Stage 4 — Build PyTorch Recognizer
    model, converter = build_recognizer(args.model, opt, device)

    # Processing Loop
    page_results = []
    for page_num, page_img in enumerate(pages, 1):
        print(f"\n[Page {page_num}/{len(pages)}]")

        # Run Stage 3: Mask Isolation & Zoom Out directly to memory array
        line_images = extract_isolated_lines(paddle_predictor, page_img)
        print(f"  Extracted {len(line_images)} clean line crops via Mask-Gated OBB Optimization.")

        if not line_images:
            page_results.append([])
            continue

        # Run Stage 4: Recognition
        line_preds = recognize_lines(
            model, converter, line_images, opt, device, batch_size=args.batch_size
        )
        page_results.append(line_preds)

        print(f"  Sample predictions (first 5):")
        for pred, conf in line_preds[:5]:
            print(f"    [{conf:.2f}] {pred}")

    # Stage 5 — Assemble Final String
    print(f"\n[Stage 5] Assembling final output text document ...")
    final_text = assemble_text(page_results, min_confidence=args.min_conf)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"\n✅ Done! OCR Output successfully saved to: {args.output}")
    compute_metrics(page_results, ground_truth_lines=None)


if __name__ == '__main__':
    main()