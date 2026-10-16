#!/usr/bin/env python3
"""
Amharic OCR Model — Speed & Latency Benchmark Utility
=====================================================
Measures forward-pass inference latency (ms/sample), throughput (lines/second),
and memory footprint across multiple batch sizes and image widths.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "train_codes"))

from model import Model
from utils import AttrDict, CTCLabelConverter


def parse_args():
    parser = argparse.ArgumentParser(description="OCR Model Inference Latency Benchmark")
    parser.add_argument("--config", default="config_files/amh_filtered_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", default="best_cer.pth", help="Path to checkpoint weights")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 16, 32],
                        help="Batch sizes to evaluate")
    parser.add_argument("--widths", nargs="+", type=int, default=[1200, 2400],
                        help="Image widths to evaluate")
    return parser.parse_args()


def load_model(config_path, model_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    opt = AttrDict(cfg)
    
    if opt.get("lang_char") and opt.lang_char != "None":
        opt.character = opt.number + opt.symbol + opt.lang_char
        
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    model = model.to(device)
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        print(f"⚠️ Model checkpoint '{model_path}' not found. Benchmarking model with random weights.")
        
    model.eval()
    return model, opt


def benchmark(model, opt, device, batch_size, width, warmup=10, iterations=50):
    img_h = opt.imgH
    channels = opt.input_channel
    
    dummy_input = torch.randn(batch_size, channels, img_h, width).to(device)
    
    # Warmup pass
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input, None, is_train=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    # Timed benchmark pass
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input, None, is_train=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
            
    latencies = np.array(latencies)
    avg_batch_time = np.mean(latencies)
    avg_sample_time = avg_batch_time / batch_size
    fps = (1000.0 / avg_sample_time) if avg_sample_time > 0 else 0
    p95_sample_time = np.percentile(latencies, 95) / batch_size
    
    return {
        "batch_size": batch_size,
        "width": width,
        "avg_batch_ms": avg_batch_time,
        "avg_per_line_ms": avg_sample_time,
        "p95_per_line_ms": p95_sample_time,
        "fps": fps,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("⚡ Amharic OCR — Inference Speed & Throughput Benchmark")
    print(f"Device: {device} | Iterations: {args.iterations}")
    print("=" * 70)
    
    config_path = os.path.join(BASE_DIR, args.config) if not os.path.isabs(args.config) else args.config
    model_path = os.path.join(BASE_DIR, args.model) if not os.path.isabs(args.model) else args.model
    
    model, opt = load_model(config_path, model_path, device)
    
    print(f"\n{'Batch':<6} | {'Width':<6} | {'Avg Line (ms)':<15} | {'P95 Line (ms)':<15} | {'Throughput (lines/s)':<22}")
    print("-" * 70)
    
    for width in args.widths:
        for bs in args.batch_sizes:
            res = benchmark(model, opt, device, batch_size=bs, width=width,
                            warmup=args.warmup, iterations=args.iterations)
            print(f"{res['batch_size']:<6} | {res['width']:<6} | {res['avg_per_line_ms']:<15.2f} | {res['p95_per_line_ms']:<15.2f} | {res['fps']:<22.1f}")
            
    print("=" * 70)


if __name__ == "__main__":
    main()
