# Amharic OCR Project Manual

## EasyOCR-Based Amharic/Ge'ez Text Recognition System

**Version:** 1.0  
**Model:** VGG16 + BiLSTM + CTC  
**Framework:** EasyOCR (deep-text-recognition-benchmark)  
**Language:** Amharic / Ge'ez (Ethiopic script)  
**Status:** On Training

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Configuration File](#5-configuration-file)
6. [How to Train](#6-how-to-train)
7. [How to Evaluate](#7-how-to-evaluate)
8. [How to Test on PDF or Image](#8-how-to-test-on-pdf-or-image)
9. [Current Model Performance](#9-current-model-performance)
10. [Bugs Fixed During Development](#10-bugs-fixed-during-development)
11. [Augmentation Pipeline](#11-augmentation-pipeline)
12. [Checkpoint Management](#12-checkpoint-management)
13. [Next Steps to Improve](#13-next-steps-to-improve)

---

## 1. Project Overview

This project trains and deploys an Amharic Optical Character Recognition (OCR) model capable of reading:

- **Synthetic** text images (rendered Amharic)
- **Typed** printed Amharic text
- **Handwritten** Amharic text
- **HDD (Historical Document Data)** — historical Ge'ez manuscripts

The model reads a cropped line-image of Amharic text and outputs the recognized Unicode string. For full-document inference, a CRAFT text detector first finds and crops each line from a page, then the recognition model reads each line.

---

## 2. Project Structure

```
OCR_ba_EasyOCR/
│
├── train.py                      ← Main training script
├── test.py                       ← Validation function (imported by train + evaluate)
├── evaluate.py                   ← Full evaluation on test set → CSV output
├── model.py                      ← OCR model architecture (VGG16+BiLSTM+CTC)
├── dataset.py                    ← Dataset loading + AmharicAugmentation class
├── utils.py                      ← CTC/Attn converters, decode functions
│
├── config_files/
│   └── amh_filtered_config.yaml  ← ★ Primary config — run this for training
    └──test_config.yaml  ← For testing
│
├── modules/
│   ├── feature_extraction.py
│   ├── sequence_modeling.py
    ├── vgg16_pretrained_text_recognition.py  ← pretrained VGG16 feature extractor
    ├── prediction.py
    ├── sequence_modeling.py
    ├── pretrained_resnet_feature_extractor.py ← pretrained ResNet feature extractor
│   └── __init__.py
│
├── craft/                        ← CRAFT text detection codebase
│   ├── model/
│   │   ├── craft.py              ← CRAFT architecture (VGG16-BN backbone)
│   │   └── vgg16_bn.py           ← VGG16-BN for CRAFT
│   ├── utils/
│   │   └── craft_utils.py        ← getDetBoxes, adjustResultCoordinates
│   └── data/
│       └── imgproc.py            ← Image preprocessing for CRAFT
│
├── saved_models/
│   └── Amharic_OCR_Model_VGG16/  ← All checkpoints saved here
│       ├── best_accuracy.pth     ← Highest exact-match accuracy checkpoint
│       ├── best_cer.pth          ← Lowest CER checkpoint (recommended for use)
│       ├── best_norm_ED.pth      ← Best normalized edit distance
│       ├── best_wer.pth          ← Best word error rate
│       ├── best_of_best.pth      ← Best composite score (acc+CER+normED)
│       └── log_train.txt         ← Full training log
│
├── amharic_pdf_ocr.py            ← ★ PDF / image inference pipeline
└── select_best_checkpoint.py     ← Evaluate all checkpoints → pick best

```

---

## 3. Dataset

| Split      | Samples         |
| ---------- | --------------- |
| Training   | ~140,000 images |
| Validation | ~14,000 images  |

**Data types included:**

- `synth` — Synthetically rendered Amharic text (various fonts)
- `typed` — Scanned printed/typed Amharic documents
- `hand` — Handwritten Amharic text
- `hdd` — Historical Ge'ez manuscript scans

**Image format:** Grayscale, resized to `64 × 1200` pixels (H × W)  
**Label format:** Unicode Amharic/Ge'ez string  
**Character vocabulary:** ~434 characters (Ethiopic fidel + digits + punctuation + Latin)

**Dataset CSV structure** (`test.csv`):

```
img_filename, img_transcription, type
synth_image_13181_1.png, ወጡ፡አየሃቡ፡የሚኖሩ, synth
hand_1000_line_4.png, ..., hand
```

---

## 4. Model Architecture

```
Input image (1 × 64 × 1200)
        ↓
VGG16 Feature Extractor   → output_channel=512
        ↓
BiLSTM Sequence Modeling   → hidden_size=256
        ↓
CTC Prediction Head        → num_class = len(character_set)
        ↓
Greedy CTC Decode
        ↓
Predicted Amharic text string
```

| Component          | Setting                                                      |
| ------------------ | ------------------------------------------------------------ |
| Transformation     | None                                                         |
| Feature Extraction | VGG16                                                        |
| Sequence Modeling  | BiLSTM                                                       |
| Prediction         | CTC                                                          |
| Input channels     | 1 (grayscale)                                                |
| Output channels    | 512                                                          |
| Hidden size        | 256                                                          |
| Image H × W        | 64 × 1200                                                    |
| Max label length   | 150 characters                                               |
| Optimizer          | Adadelta (lr=1.0) #(currently changed to AdamW (lr=0.00001)) |
| Batch size         | 32                                                           |

---

## 5. Configuration File

**Primary config:** `config_files/amh_filtered_config.yaml`

Key settings to know before running:

```yaml
experiment_name: "Amharic_OCR_Model_VGG16" # folder name for saved checkpoints

# ── Dataset paths (edit these to your local paths) ──
train_data: "../../../Datasets/CAC_DA_Amh_OCR/train"
valid_data: "../../../Datasets/CAC_DA_Amh_OCR/val"

# ── Training ──
num_iter: 200000 # total training iterations (currently extended to 280,000)
valInterval: 1000 # validate every 1000 iterations
batch_size: 32

# ── Image ──
imgH: 64
imgW: 1200
PAD: True # pad shorter images to imgW

# ── Augmentation ──
augment: True # set False to disable during evaluation
augment_prob: 0.4 # 0.4=light, 0.6=medium

# ── Architecture ──
FeatureExtraction: "VGG"
SequenceModeling: "BiLSTM"
Prediction: "CTC"

# ── Resume training ──
saved_model: "" # set to best_cer.pth path to resume
```

---

## 6. How to Train

### First-time training:

```bash
cd OCR_bg_EasyOCR
python train.py --config config_files/amh_filtered_config.yaml
```

### Resume training after a crash:

```bash
python train.py \
  --config config_files/amh_filtered_config.yaml \
  --saved_model saved_models/Amharic_OCR_Model_VGG16/best_cer.pth
```

### What gets saved during training:

| File                | When saved                                      | What it is                                                                      |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------- |
| `best_accuracy.pth` | When accuracy improves                          | Weights at highest exact-match accuracy                                         |
| `best_cer.pth`      | When CER improves                               | Weights at lowest character error rate                                          |
| `best_norm_ED.pth`  | When normED improves                            | Weights at best normalized edit distance                                        |
| `best_of_best.pth`  | Every valInterval (if composite score improves) | Weights with best combined score: `(1−CER)×0.50 + (acc/100)×0.30 + normED×0.20` |

---

## 7. How to Evaluate

### Full evaluation on test set:

```bash
python evaluate.py \
  --config config_files/test_config.yaml \
  --model  saved_models/Amharic_OCR_Model_VGG16/best_cer.pth
```

**Output files:**

- `saved_models/test_evaluations/<name>/predictions/test_predictions.csv` → per-sample predictions with CER/WER
- `saved_models/test_evaluations/<name>/metrics/test_metrics.txt` → accuracy breakdown by data type (synth/typed/hand/hdd)

### Select the best checkpoint after training completes:

```bash
python select_best_checkpoint.py \
  --config         config_files/amh_filtered_config.yaml \
  --experiment_dir saved_models/Amharic_OCR_Model_VGG16
```

This evaluates all `best_*.pth` files on your validation set, ranks them by composite score, and saves the winner as `best_of_best.pth`.

## 8. How to Test on PDF or Image

### Setup (first time only):

```bash
pip install pdf2image easyocr opencv-python
```

For PDF support on Windows, install Poppler and update the `DEFAULT_POPPLER` path in the script.

### Quick test on a single image:

```bash
python amharic_pdf_ocr.py \
  --input  "path/to/image.jpg" \
  --model  "saved_models/Amharic_OCR_Model_VGG16/best_cer.pth" \
  --output "result.txt"
```

### Test on a PDF:

```bash
python amharic_pdf_ocr.py \
  --input  "path/to/document.pdf" \
  --model  "saved_models/Amharic_OCR_Model_VGG16/best_cer.pth" \
  --output "result.txt" \
  --pages  "1"
```

### Pipeline stages inside `amharic_pdf_ocr.py`:

| Stage | What it does                                                             | Code function                           |
| ----- | ------------------------------------------------------------------------ | --------------------------------------- |
| 1     | PDF → page images at 300 DPI (or load image directly)                    | `load_input()`                          |
| 2     | Load Paddle Inference engine on CPU & remove bugged oneDNN passes        | `build_paddle_detector()`               |
| 3     | Extract line heatmaps and filter out tiny noise contours                 | `extract_isolated_lines()`              |
| 4     | Build rotated bounding boxes with extra padding to prevent text clipping | `extract_isolated_lines()`              |
| 5     | Mask out adjacent lines to isolate one target text line at a time        | `extract_isolated_lines()`              |
| 6     | Flatten and deskew each cropped line via perspective transformation      | `deskew_and_crop()`                     |
| 7     | Load VGG-BiLSTM network, batch line images, and recognize Amharic text   | `build_recognizer(), recognize_lines()` |
| 8     | Assemble text using line breaks (\n) and print confidence metrics        | `assemble_text(), compute_metrics()`    |

**Tuning parameters:**

```bash
--merge_ratio 0.6   # line merging: lower=more merging, higher=less merging
--min_conf    0.3   # lines below this confidence marked [? text ?]
--dpi         300   # PDF rendering DPI (increase to 400 for small text)
```

---

## 9. Current Model Performance

### Latest Training Run (with Augmentation + Fixed Metrics)

**Status: 75,000 / 280,000 iterations (26.8% complete)**

> This run includes: `AmharicAugmentation` pipeline, fixed `test.py` (trailing whitespace stripped before scoring), and optimizer-state resumable checkpoints. Metrics are directly comparable and correct.

#### Metrics at each validation checkpoint:

| Iteration  | Valid Loss  | Accuracy    | CER        | WER        | Norm ED    | Char F1    | Train Loss |
| ---------- | ----------- | ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 50,000     | 0.18684     | 55.668%     | 0.0815     | 0.4017     | 0.9227     | 0.9073     | 0.08768    |
| 55,000     | 0.19091     | 55.512%     | 0.0816     | 0.4035     | 0.9227     | 0.9088     | 0.08477    |
| **60,000** | **0.18899** | **55.717%** | **0.0812** | **0.4000** | **0.9229** | **0.9085** | 0.08214    |
| 65,000     | 0.19092     | 55.610%     | 0.0819     | 0.4014     | 0.9224     | 0.9077     | 0.08504    |
| 70,000     | 0.18851     | 55.717%     | 0.0812     | 0.4012     | 0.9231     | 0.9094     | 0.08356    |
| 75,000     | 0.18926     | 55.726%     | 0.0814     | 0.4008     | 0.9229     | 0.9087     | 0.08139    |

#### Best metrics recorded so far:

| Metric                 | Best Value  | At iteration    |
| ---------------------- | ----------- | --------------- |
| Accuracy (exact match) | **55.726%** | 75,000          |
| CER (char error rate)  | **0.0812**  | 60,000 / 70,000 |
| WER (word error rate)  | **0.4000**  | 60,000          |
| Norm ED                | **0.9231**  | 70,000          |
| Char Precision         | **0.9106**  | 70,000          |
| Char Recall            | **0.9082**  | 70,000          |
| Char F1                | **0.9094**  | 70,000          |
| CTC Loss (valid)       | **0.18090** | ~45,000         |

#### Sample predictions at 75,000 iterations:

| Ground Truth           | Prediction             | Conf  | Correct     |
| ---------------------- | ---------------------- | ----- | ----------- |
| ደውኪ፡ይትመተራ፡ኦሔዋ          | ደውኪ፡ይትመተራ፡ሐዋ           | 0.982 | ✗           |
| ግንቦት ቀን ዓ ም በመቻሬ ሜዳ... | ግንቦት ቀን ዓ ም በመቻሬ ሜዳ... | 0.474 | ✓           |
| ቀ፡መላእክት                | ቀ፡መላእክት፡               | 0.996 | ✗ (extra ፡) |

---

### Comparison: Previous Run vs Current Run

| Metric     | Previous run @ 91k | Current run @ 75k | Improvement       |
| ---------- | ------------------ | ----------------- | ----------------- |
| Valid Loss | 0.46452            | **0.18926**       | **−59% lower**    |
| Accuracy   | 54.190%            | **55.726%**       | **+1.5%**         |
| CER        | 0.0891             | **0.0812**        | **−8.9% lower**   |
| WER        | 0.4199             | **0.4008**        | **−4.5% lower**   |
| Norm ED    | 0.9162             | **0.9231**        | **+0.69% higher** |
| Char F1    | 0.9033             | **0.9094**        | **+0.61% higher** |

The current run is significantly better across every metric despite being at fewer iterations. The valid loss dropped from ~0.44 to ~0.19 — this is the combined effect of the augmentation pipeline (making the model generalize better) and the corrected `test.py` metrics showing real numbers.

---

### Training trend analysis

Valid loss is oscillating in a narrow band (0.188–0.191) while accuracy and CER remain nearly flat (±0.002) over the last 25,000 iterations. This indicates:

- The model is in a **stable plateau** — not degrading, not improving rapidly
- The optimizer (Adadelta) has found a local optimum at the current learning rate
- **Expected behavior at 26% of training** — breakthroughs often occur later as the optimizer adapts

**Key observations from sample predictions:**

- Long typed/printed lines: recognized nearly perfectly (`...` — 133 chars, exact match ✓)
- Short Ge'ez/HDD lines: mostly correct with occasional character confusion (`ኦሔዋ→ሐዋ`, `ወነጸ→ወነድ`)
- Trailing punctuation sometimes appended incorrectly (`ቀ፡መላእክት` → `ቀ፡መላእክት፡`) — common CTC artifact
- Very low confidence on long correct predictions (e.g., 0.0023 confidence on a perfect 88-char match) — this is expected CTC behavior where the model distributes probability across many valid alignment paths

---

## 10. Bugs Fixed During Development

These bugs were found and fixed during development. All fixes are in the current codebase.

### Bug 1 — CER inflation from trailing spaces (CRITICAL)

**File:** `test.py`  
**Problem:** The CTC greedy decoder was outputting trailing space characters from the padding region of each prediction. These spaces were not stripped before computing metrics, making every prediction string longer than the actual text, inflating CER and deflating accuracy.  
**Impact:** Test set accuracy reported as 31.59% was actually 41.10%. CER reported as 0.1857 was actually 0.1158.  
**Fix:** Added `pred.rstrip()` before all metric computation and moved `all_preds` collection inside the per-sample loop so the CSV also receives clean strings.

### Bug 2 — `all_preds` collected before stripping

**File:** `test.py`  
**Problem:** Even after fixing Bug 1, the `all_preds` list (used by `evaluate.py` to write the CSV) was populated from the raw unstripped `preds_str` list outside the per-sample loop. So the CSV CER column was still wrong even though the summary metrics were correct.  
**Fix:** Moved `all_preds.append(pred)` inside the per-sample loop, after stripping.

### Bug 3 — No optimizer state saved in checkpoints

**File:** `train.py`  
**Problem:** Only `model.state_dict()` was saved. Resuming training restarted the Adadelta optimizer's running averages from zero, causing a loss spike after every resume.  
**Fix:** `(checkpoints like best_cer..).pth` now saves both `model_state_dict` and `optimizer_state_dict`. Loading code detects old vs new format for backward compatibility.

---

## 11. Augmentation Pipeline

**File:** `dataset.py` — class `AmharicAugmentation`  
Applied only during training (`is_train=True`). Never applied to validation or test data.

Designed from pixel analysis of actual training samples:

| Transform                    | Probability | Why                                     |
| ---------------------------- | ----------- | --------------------------------------- |
| Rotation ±2°                 | 55%         | Document skew (all types)               |
| Shear ±2°                    | 35%         | Italic lean (typed/handwritten)         |
| Elastic distortion           | 45%         | Pen pressure variation (handwritten)    |
| Morphological (dilate/erode) | 55%         | Faded/thick ink (HDD/Ge'ez)             |
| Salt & pepper noise          | 45%         | Scanner dust (HDD scans)                |
| Low-resolution upscale       | 35%         | Low-DPI digitization (HDD)              |
| JPEG compression             | 50%         | Compression artifacts (typed/synthetic) |
| Gaussian noise σ=3–15        | 50%         | Scan grain (all types)                  |
| Gaussian blur r=0.3–1.2      | 40%         | Out-of-focus scanning                   |
| Motion blur                  | 18%         | Camera shake                            |
| Brightness ±30%              | 60%         | Faded ink / over-exposed scans          |
| Contrast ±30%                | 60%         | Low-contrast HDD documents              |
| Shadow stripe                | 20%         | Page fold / scanner illumination        |

**To adjust:** Edit `augment_prob` in `amh_filtered_config.yaml`:

- `0.4` = light (current, safe for training)
- `0.6` = medium (more aggressive, for robustness)
- `False` for `augment` = disabled

---

## 12. Checkpoint Management

### Which checkpoint to use for what:

| Purpose                     | Checkpoint                      |
| --------------------------- | ------------------------------- |
| Inference / deployment      | `best_cer.pth`                  |
| Resume training after crash | `best_cer.pth`                  |
| Compare model quality       | Run `select_best_checkpoint.py` |

### Why `best_cer.pth` over `best_accuracy.pth`:

Accuracy is exact-match — one wrong character fails the whole line. CER measures character-level quality and gives partial credit. For complex Ethiopic fidel with ~434 characters and many visually similar pairs (መ/ወ/, ሐ/ሓ, ሰ/ለ, etc.), CER is the more informative metric for real-world performance.

---

## 13. Next Steps to Improve

### Immediate (while current training continues):

1. **Wait for training to complete** — iterations remain.
2. **Run `select_best_checkpoint.py`** once training ends to confirm `best_of_best.pth` is truly the best checkpoint across all metrics.
3. **Test on real documents** using `amharic_pdf_ocr.py` at each major milestone (every 50k iterations) to track real-world improvement.

### Model improvements:

5. **Increase `num_iter`** — 280,000 may not be enough for this character set size. Consider extending to 400,000–500,000 if the model is still improving at 280k.
6. **TPS (Thin Plate Spline) transformation** — change `Transformation: "None"` to `"TPS"` in config for a new training run. TPS adds spatial rectification that handles curved/distorted text lines better, especially important for handwritten data.

### Data improvements:

7. **More handwritten data** — the model still struggles most on handwritten text. Collecting and annotating more handwritten Amharic samples (especially diverse writers and pen types) would improve this sub-domain significantly.
8. **HDD data quality** — historical Ge'ez documents are the hardest data type. Consider pre-processing HDD images with binarization and denoising before training rather than relying on augmentation alone.
9. **Train CRAFT on Amharic documents** — the `craft/train.py` codebase is already in the project. Training a CRAFT model specifically on annotated Amharic document pages would significantly improve text detection quality for the full PDF pipeline, especially for tightly-spaced handwriting.

### Architecture improvements (next training run):

10. **ResNet34 vs VGG16 comparison** — a parallel training run with `FeatureExtraction: "ResNet"` to determine which backbone performs better on Amharic fidel. VGG16 was confirmed superior in early experiments but a full-length comparison at the correct iteration count has not been done.
11. **Attention decoder** — switch `Prediction: "CTC"` to `"Attn"` for a new experiment. Attention-based decoding handles variable-length sequences better and may reduce WER on longer lines.

---

## Quick Reference

### Most important files:

```bash
# Train
python train.py --config config_files/amh_filtered_config.yaml

# Evaluate on test set
python evaluate.py --config config_files/test_config.yaml \
                   --model saved_models/Amharic_OCR_Model_VGG16/best_cer.pth

# Run on PDF or image
python amharic_pdf_ocr.py \
  --input  document.pdf \
  --model  saved_models/Amharic_OCR_Model_VGG16/best_cer.pth \
  --output result.txt

# Select best checkpoint
python select_best_checkpoint.py \
  --experiment_dir saved_models/Amharic_OCR_Model_VGG16

```

### Current best model location:

```
saved_models/Amharic_OCR_Model_VGG16/best_cer.pth
saved_models/Amharic_OCR_Model_VGG16/best_of_best.pth
```

### Key metric to watch during training:

**CER** (Current_CER in training log) — lower is better. Current best: **0.0812**

---

_Manual prepared based on project development history. Last updated at training 75k/280k iteration._
