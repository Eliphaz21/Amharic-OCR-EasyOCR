import os
import sys
import argparse
import pandas as pd
import time
import yaml
import csv
import torch
import torch.utils.data
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model
from test import validation


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    opt = Namespace(**cfg)
    return opt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--config', type=str, default='config_files/test_config.yaml',
                        help='Path to test config yaml file')
    parser.add_argument('--model', type=str, default=None, help='Path to saved model checkpoint')
    parser.add_argument('--test_data', type=str, default=None, help='Path to evaluation test dataset')
    parser.add_argument('--show_number', type=int, default=5, help='Number of sample predictions to print')
    return parser.parse_args()


def print_metrics(results, opt):
    print("================================================================================")
    print(f"Test samples: {results[14]}")
    print(f"Valid loss:   {results[0]:.5f}")
    if 'CTC' in opt.Prediction:
        print(f"CTC loss:     {results[15]:.5f}")
    if 'Attn' in opt.Prediction:
        print(f"CE loss:      {results[16]:.5f}")
    print(f"Accuracy:     {results[1]:.3f}%")
    print(f"Norm ED:      {results[2]:.4f}")
    print(f"WER:          {results[3]:.4f}")
    print(f"CER:          {results[4]:.4f}")
    print(f"SER:          {results[5]:.2f}%")
    print(f"Char P/R/F1:  {results[6]:.4f} / {results[7]:.4f} / {results[8]:.4f}")
    print(f"Latency:      {results[9]:.2f} ms/sample")
    print(f"Total infer:  {results[13]:.2f} s")
    print("================================================================================")


def show_samples(labels, preds, confidence, opt, n=5):
    print("\nSample predictions:")
    print("-" * 80)
    print(f"{'Ground Truth':<30} | {'Prediction':<30} | {'Conf':<6} | {'OK'}")
    print("-" * 80)
    for i, (gt, pred, conf) in enumerate(zip(labels[:n], preds[:n], confidence[:n])):
        if 'Attn' in opt.Prediction:
            gt = gt[:gt.find('[s]')] if '[s]' in gt else gt
            pred = pred[:pred.find('[s]')] if '[s]' in pred else pred
        ok = (gt == pred)
        print(f"{gt:<30} | {pred:<30} | {conf:.4f} | {ok}")
    print("-" * 80)


def compute_cer_wer(gt, pred):
    """
    Per-sample CER and WER using edit distance.

    CER = character-level edit distance / number of ground-truth characters
    WER = word-level edit distance / number of ground-truth words

    Edge case: if ground truth is empty, define CER/WER as 0.0 when the
    prediction is also empty (perfect match) and 1.0 otherwise (penalize
    any hallucinated output against an empty reference).
    """
    gt_len = len(gt)
    cer = edit_distance(pred, gt) / gt_len if gt_len > 0 else (0.0 if len(pred) == 0 else 1.0)

    gt_words = gt.split()
    pred_words = pred.split()
    wer_len = len(gt_words)
    wer = edit_distance(pred_words, gt_words) / wer_len if wer_len > 0 else (0.0 if len(pred_words) == 0 else 1.0)

    return cer, wer


def save_results(results, opt, output_dir):
    import pandas as pd
    import os

    # ORIGINAL_CSV_PATH = "../../../Datasets/OCR/CAC_DA_Amh_OCR/test.csv"
    ORIGINAL_CSV_PATH = "C:/Users/PC/Desktop/CAC_DA_Amh_OCR/test.csv"


    metrics_dir = os.path.join(output_dir, 'metrics')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    all_preds, all_labels, all_confidence = results[17:20]

    lookup_df = None
    try:
        lookup_df = pd.read_csv(ORIGINAL_CSV_PATH, encoding='utf-8-sig', engine='python')
        print(f"📖 SUCCESS: Loaded original mapping index from {ORIGINAL_CSV_PATH}")

        # Normalize the transcriptions inside the lookup frame by stripping all spaces, newlines, and carriage returns
        lookup_df['clean_transcription'] = lookup_df['img_transcription'].astype(str).str.strip().str.replace(r'\s+',
                                                                                                              '',
                                                                                                              regex=True)
    except Exception as e:
        print(f"⚠️ CRITICAL ERROR: Could not read CSV file at {ORIGINAL_CSV_PATH}. Reason: {e}")

    rows = []

    for idx, (gt, pred, conf) in enumerate(zip(all_labels, all_preds, all_confidence)):
        if 'Attn' in opt.Prediction:
            gt_clean = gt[:gt.find('[s]')] if '[s]' in gt else gt
            pred_clean = pred[:pred.find('[s]')] if '[s]' in pred else pred
        else:
            gt_clean = gt
            pred_clean = pred

        # Setup standard unique fallbacks if string matching ever drops out
        img_name = f"unmapped_index_{idx:06d}"
        img_type = "not_found_in_csv"

        if lookup_df is not None:
            # Clean the model evaluation ground truth target string using the identical method
            gt_normalized = str(gt_clean).strip().replace(" ", "")

            # Find the row match
            matched_records = lookup_df[lookup_df['clean_transcription'] == gt_normalized]

            if not matched_records.empty:
                first_match = matched_records.iloc[0]

                # 🎯 EXACT ASSIGNMENT: Extracts the exact image filename from your CSV without modifications
                img_name = str(first_match.get('img_filename', img_name)).strip()

                # 🎯 EXACT ASSIGNMENT: Pulls the raw classification text string directly
                img_type = str(first_match.get('type', 'unknown')).strip()

        # Per-sample CER / WER
        cer, wer = compute_cer_wer(gt_clean, pred_clean)

        rows.append({
            'image_name': img_name,  # ✅ Contains authentic filenames like 'synth_image_12990_0.png'
            'image_type': img_type,  # ✅ Contains authentic raw classes like 'synth'
            'ground_truth': gt_clean,
            'prediction': pred_clean,
            'cer': cer,
            'wer': wer,
            'confidence': float(conf),
            'correct': pred_clean == gt_clean,

        })

    # Save to your output tracking file
    predictions_df = pd.DataFrame(rows)
    csv_path = os.path.join(predictions_dir, 'test_predictions.csv')
    predictions_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Metric compilation breakdown loop
    log_path = os.path.join(metrics_dir, 'test_metrics.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'model: {opt.saved_model}\n')
        f.write(f'samples: {results[14]}\n')
        f.write(f'accuracy: {results[1]:0.3f}%\n\n')

        f.write('IMAGE SUB-DOMAIN TYPE BREAKDOWN SUMMARY (DIRECT MATCHING)\n')
        f.write('=' * 80 + '\n')

        # Reads the type allocations dynamically from whatever labels existed in the csv column
        unique_types = predictions_df['image_type'].unique()
        for category in unique_types:
            sub_df = predictions_df[predictions_df['image_type'] == category]
            total_samples = len(sub_df)
            correct_samples = sub_df['correct'].sum()
            accuracy = (correct_samples / total_samples) * 100 if total_samples > 0 else 0
            avg_cer = sub_df['cer'].mean() if total_samples > 0 else 0
            avg_wer = sub_df['wer'].mean() if total_samples > 0 else 0

            f.write(f'Sub-Domain Category: [{category}]\n')
            f.write(f'  - Evaluated Samples    : {total_samples}\n')
            f.write(f'  - Correctly Recognized : {correct_samples} / {total_samples}\n')
            f.write(f'  - Domain Line Accuracy : {accuracy:0.2f}%\n')
            f.write(f'  - Domain Avg CER       : {avg_cer:0.4f}\n')
            f.write(f'  - Domain Avg WER       : {avg_wer:0.4f}\n')
            f.write('-' * 40 + '\n')

    print(f'\nSaved predictions folder to {csv_path}')
    print(f'Saved metrics folder to {log_path}')
    return csv_path, log_path
def evaluate(opt, output_dir=None, show_number=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    print(f"🔄 Loading strict weights from {opt.saved_model}...")
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()

    AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_dataset, _ = hierarchical_dataset(root=opt.test_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True
    )
    print(f"[Dataset Log] Loaded dataset from {opt.test_data} with {len(eval_dataset)} images.")
    print(f"Evaluating {len(eval_dataset)} test samples from {opt.test_data}")

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(
        device) if 'CTC' in opt.Prediction else torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    with torch.no_grad():
        results = validation(model, criterion, evaluation_loader, converter, opt, device, collect_all=True)

    print_metrics(results, opt)

    # Using specific positions safely
    show_samples(results[18], results[17], results[19], opt, n=show_number)

    if output_dir:
        save_results(results, opt, output_dir)


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.config):
        print(f'Config not found: {args.config}')
        sys.exit(1)

    opt = get_config(args.config)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if 'character' not in cfg or not cfg['character']:
        print("❌ Error: 'character' string is missing or empty in your configuration file!")
        print(f"Please open '{args.config}' and add your character vocabulary set.")
        sys.exit(1)

    if cfg.get('character'):
        opt.character = cfg['character']

    if args.model:
        opt.saved_model = args.model
    elif cfg.get('saved_model'):
        opt.saved_model = cfg['saved_model']

    if args.test_data:
        opt.test_data = args.test_data
    elif cfg.get('test_data'):
        opt.test_data = cfg['test_data']
    else:
        # ── 🌟 FIX: Safe attribute check fallback configuration ───────────
        opt.test_data = getattr(opt, 'valid_data', None)
        if not opt.test_data:
            print("❌ Error: No test dataset path was provided via command arguments or config file.")
            sys.exit(1)

    experiment_name = cfg.get('experiment_name', 'test_evaluation')
    dynamic_output_dir = os.path.join('saved_models', 'test_evaluations', experiment_name)
    print(f"📂 Execution Directory: {dynamic_output_dir}")

    evaluate(opt, output_dir=dynamic_output_dir, show_number=args.show_number)