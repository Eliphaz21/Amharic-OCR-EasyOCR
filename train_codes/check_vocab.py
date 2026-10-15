#!/usr/bin/env python3
"""
Amharic Vocabulary & CSV Label Verifier
=======================================
Inspects dataset CSVs against the configured Ethiopic character vocabulary
to detect Out-Of-Vocabulary (OOV) characters, corrupt Unicode, or formatting errors.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Amharic OCR Vocabulary Verifier")
    parser.add_argument("--config", default="config_files/amh_filtered_config.yaml",
                        help="Path to YAML config containing 'number', 'symbol', 'lang_char'")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV file (e.g. train.csv, val.csv)")
    parser.add_argument("--max_length", type=int, default=150, help="Max allowed label character length")
    parser.add_argument("--output_clean", default=None, help="Optional path to export filtered/cleaned CSV")
    return parser.parse_args()


def load_vocab_from_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "character" in cfg and cfg["character"]:
        char_set = set(cfg["character"])
    else:
        number = cfg.get("number", "")
        symbol = cfg.get("symbol", "")
        lang_char = cfg.get("lang_char", "")
        char_set = set(number + symbol + lang_char)

    return char_set


def inspect_dataset(csv_path, vocab_set, max_length=150, clean_output_path=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"📖 Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python", keep_default_na=False)

    if "img_transcription" not in df.columns:
        raise ValueError("CSV must contain 'img_transcription' column.")

    total_samples = len(df)
    oov_counter = Counter()
    exceeded_length_count = 0
    empty_label_count = 0
    clean_rows = []

    for idx, row in df.iterrows():
        text = str(row["img_transcription"]).strip()
        filename = str(row.get("img_filename", f"row_{idx}"))

        if len(text) == 0:
            empty_label_count += 1
            continue

        if len(text) > max_length:
            exceeded_length_count += 1
            continue

        row_oov = [ch for ch in text if ch not in vocab_set]
        if row_oov:
            oov_counter.update(row_oov)
            continue

        clean_rows.append(row)

    valid_samples = len(clean_rows)
    filtered_out = total_samples - valid_samples

    print("\n" + "=" * 65)
    print("DATASET VOCABULARY AUDIT REPORT")
    print("=" * 65)
    print(f"Total rows inspected      : {total_samples}")
    print(f"Valid compliant rows       : {valid_samples} ({valid_samples / total_samples * 100:.2f}%)")
    print(f"Filtered out rows          : {filtered_out} ({filtered_out / total_samples * 100:.2f}%)")
    print(f"  - Empty labels           : {empty_label_count}")
    print(f"  - Exceeded length (>{max_length}): {exceeded_length_count}")
    print(f"  - Contained OOV chars    : {filtered_out - empty_label_count - exceeded_length_count}")
    print("=" * 65)

    if oov_counter:
        print("\nTop Out-Of-Vocabulary (OOV) Characters Detected:")
        print(f"{'Char':<6} | {'Unicode Hex':<12} | {'Frequency':<10}")
        print("-" * 35)
        for char, count in oov_counter.most_common(25):
            u_hex = f"U+{ord(char):04X}"
            print(f"{char:<6} | {u_hex:<12} | {count:<10}")
        print("\n💡 Recommendation: Add these characters to 'lang_char' or 'symbol' in your config if legitimate.")
    else:
        print("\n✅ Perfect match! All characters in this dataset exist in the configuration vocabulary.")

    if clean_output_path:
        clean_df = pd.DataFrame(clean_rows)
        clean_df.to_csv(clean_output_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Saved sanitized CSV ({valid_samples} rows) to: {clean_output_path}")


def main():
    args = parse_args()
    vocab = load_vocab_from_config(args.config)
    print(f"Loaded {len(vocab)} vocabulary characters from {args.config}")
    inspect_dataset(args.csv, vocab, max_length=args.max_length, clean_output_path=args.output_clean)


if __name__ == "__main__":
    main()
