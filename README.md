# Amharic-OCR-EasyOCR

Optical Character Recognition (OCR) system specialized for **Amharic / Ge'ez (Ethiopic)** script, built on top of the **Deep Text Recognition Benchmark** (EasyOCR backbone) and integrated with **PaddleOCR DBNet** for document line segmentation.

Supports multi-domain text recognition:
- **Synthetic**: Computer-rendered Ethiopic fonts
- **Typed**: Scanned printed books, letters, and forms
- **Handwritten**: Freeform handwritten Amharic text
- **Historical Document Data (HDD)**: Ancient Ge'ez manuscript scans with faded ink and parchment aging

---

## Architecture Overview

```
Input Page (PDF / Image at 300 DPI)
               │
               ▼
   [ PaddleOCR DBNet Detector ]        ← Workaround for Intel oneDNN crashes
               │
               ▼
   [ Mask-Gated Line Isolation ]       ← Deskew & perspective warp with zoom-out padding
               │
               ▼
   [ VGG16 Backbone (Modified) ]       ← MaxPool kernels adapted to preserve sequence width
               │
               ▼
   [ 2-Layer BiLSTM Sequence Head ]    ← Dropout 0.2 regularization
               │
               ▼
   [ CTC Prediction & Greedy Decode ]  ← Vocabulary: ~435 Ethiopic fidel, numerals & symbols
               │
               ▼
       Recognized Text File
```

---

## Performance Summary

| Metric | Current Value | Notes |
| :--- | :--- | :--- |
| **Character Error Rate (CER)** | **0.0812** | Lowest error checkpoint (`best_cer.pth`) |
| **Exact Line Accuracy** | **55.73%** | Line exact match |
| **Normalized Edit Distance** | **0.9231** | ICDAR 2019 metric |
| **Character F1-Score** | **0.9094** | Precision 0.9106 / Recall 0.9082 |
| **Word Error Rate (WER)** | **0.4000** | Word-level Levenshtein distance |

---

## Repository Structure

```
├── Amharic_OCR_Project_Manual.md      # Full technical manual and training reports
├── README.md                          # Repository overview
├── .gitignore                         # Binary & artifact exclusion rules
│
├── train_codes/                       # Training pipeline
│   ├── train.py                       # Main training loop with mixed precision (AMP)
│   ├── test.py                        # Validation & metric evaluation engine
│   ├── model.py                       # CRNN model definition (VGG + BiLSTM + CTC)
│   ├── dataset.py                     # Dataset loading & Albumentations augmentations
│   ├── utils.py                       # CTC & Attention converters, beam search
│   ├── config_files/                  # Training hyperparameters & character sets
│   ├── craft/                         # CRAFT text detector subpackage
│   └── modules/                       # Backbone neural network modules
│
└── test_codes/                        # Testing, evaluation & document OCR
    ├── Amharic pdf ocr.py             # End-to-end PDF/image inference pipeline
    ├── crop_img.py                    # Mask-gated line cropper with perspective deskew
    ├── evaluate.py                    # Test set evaluation with CSV & domain breakdown
    ├── custom_tracker/                # PaddleOCR DBNet inference models
    └── best_cer.pth                   # Pre-trained Amharic recognition model weights
```

---

## Quickstart

### 1. Installation

```bash
pip install torch torchvision albumentations opencv-python pillow pandas pyyaml natsort nltk pdf2image paddlepaddle
```

*(For PDF processing on Windows, install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) and ensure it is in your system PATH).*

### 2. Recognize Text from a PDF or Image

```bash
cd test_codes
python "Amharic pdf ocr.py" --input "path/to/document.pdf" --model "best_cer.pth" --output "output.txt" --dpi 300
```

### 3. Evaluate Checkpoint on Test Set

```bash
cd test_codes
python evaluate.py --config config_files/test_config.yaml --model best_cer.pth
```

### 4. Train from Scratch or Resume

```bash
cd train_codes
python train.py --config config_files/amh_filtered_config.yaml --saved_model saved_models/Amharic_OCR_Model_VGG16/best_cer.pth
```

---

## Detailed Documentation

For full architecture details, bug fix histories, augmentation benchmarks, and technical walkthroughs, see the comprehensive [Amharic OCR Project Manual](Amharic_OCR_Project_Manual.md).

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
