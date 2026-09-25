import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model


def extract_image_metadata(dataset):
    """ Safely extracts image filenames and types from the evaluation dataset """

    def get_meta_from_sub(sub_ds):
        # Scan common internal dataset attributes for file lists
        for attr in ['image_path_list', 'data_line_list', 'samples', 'imgs', 'filenames']:
            if hasattr(sub_ds, attr):
                lst = getattr(sub_ds, attr)
                if lst and isinstance(lst, list):
                    return lst
        return None

    # Handle single or ConcatDataset architectures
    sub_datasets = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
    if hasattr(dataset, 'dataset_list'):
        sub_datasets = dataset.dataset_list

    all_paths = []
    for s_ds in sub_datasets:
        paths = get_meta_from_sub(s_ds)
        if paths:
            all_paths.extend(paths)
        else:
            # Fallback label generation if it's a raw database backend (e.g., LMDB)
            n_samples = len(s_ds) if hasattr(s_ds, '__len__') else 0
            if hasattr(s_ds, 'nSamples'):
                n_samples = int(s_ds.nSamples)
            all_paths.extend([f"image_{i:06d}.png" for i in range(n_samples)])

    metadata = []
    for p in all_paths:
        if isinstance(p, tuple) and len(p) > 0:
            p = p[0]
        if isinstance(p, str):
            p = p.split('\t')[0].split(',')[0].strip()
            base = os.path.basename(p)
            name, ext = os.path.splitext(base)

            # ── 🌟 Track Sub-Domain Category from Path ──────────────────────
            p_lower = p.lower()
            if 'handwritten' in p_lower:
                category = 'handwritten'
            elif 'typed' in p_lower:
                category = 'typed'
            elif 'synth' in p_lower:
                category = 'synth_image'
            elif 'hdd_train' in p_lower or 'hdd' in p_lower:
                category = 'hdd_train'
            else:
                category = ext.lstrip('.') if ext else 'unknown'

            metadata.append((name, category))
        else:
            metadata.append((str(p), 'unknown'))
    return metadata


def validation(model, criterion, evaluation_loader, converter, opt, device, collect_all=False):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    total_WER = 0
    total_CER = 0
    total_SER = 0
    char_TP = 0
    char_FP = 0
    char_FN = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    # separate averagers — one per loss type so each can be reported independently
    ctc_loss_avg = Averager()  # CTC path only
    ce_loss_avg = Averager()  # Attention (CrossEntropy) path only
    all_preds = []
    all_labels = []
    all_confidence = []

    # New collections for image metadata tracking
    all_image_names = []
    all_image_types = []

    # Advanced breakdown collections
    all_losses = []
    all_wers = []
    all_cers = []

    image_metadata = extract_image_metadata(evaluation_loader.dataset)

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        batch_start_idx = length_of_data
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)

            # CTC loss — accumulated independently for per-loss reporting
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss.to(device)
            )
            ctc_loss_avg.add(cost)

            # Calculate individual sample losses for the breakdown
            criterion_none = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
            sample_costs = criterion_none(
                preds.log_softmax(2).permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss.to(device)
            )
            batch_losses = sample_costs.detach().cpu().numpy().tolist()

            if opt.decode == 'greedy':
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
            elif opt.decode == 'beamsearch':
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        else:
            # ── Attention branch ──────────────────────────────────────────────
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # strip leading [GO] token

            # CrossEntropyLoss — same criterion object passed in from train.py
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1)
            )
            ce_loss_avg.add(cost)

            # Calculate individual sample losses for Attention sequence mapping
            criterion_none = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0).to(device)
            sample_costs = criterion_none(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1)
            )
            batch_losses = sample_costs.view(batch_size, -1).mean(dim=1).detach().cpu().numpy().tolist()
            # ─────────────────────────────────────────────────────────────────

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # ── confidence & per-sample metrics ──────────────────────────────────
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        for idx_in_batch, (gt, pred, pred_max_prob) in enumerate(zip(labels, preds_str, preds_max_prob)):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

            # Strip trailing whitespace artifacts from CTC's padding-region
            # flicker (blank/space alternation) before any metric is computed.
            # rstrip (not strip) — only trailing chars are affected, so we can
            # safely truncate pred_max_prob to match without touching leading content.
            pred_len_before_strip = len(pred)
            pred = pred.rstrip()
            gt = gt.rstrip()
            if len(pred) < pred_len_before_strip:
                pred_max_prob = pred_max_prob[:len(pred)]

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # WER
            gt_words = gt.split()
            pred_words = pred.split()
            wer = edit_distance(pred_words, gt_words) / len(gt_words) if len(gt_words) > 0 else 0
            total_WER += wer

            # CER
            cer = edit_distance(pred, gt) / len(gt) if len(gt) > 0 else 0
            total_CER += cer

            # SER
            if pred != gt:
                total_SER += 1

            # Character-level Precision / Recall / F1
            for char_idx in range(max(len(gt), len(pred))):
                gt_char = gt[char_idx] if char_idx < len(gt) else None
                pred_char = pred[char_idx] if char_idx < len(pred) else None

                if gt_char == pred_char and gt_char is not None:
                    char_TP += 1
                elif pred_char is not None and gt_char != pred_char:
                    char_FP += 1

                if gt_char is not None and gt_char != pred_char:
                    char_FN += 1

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

            # Map global index to fetch specific image details
            global_idx = batch_start_idx + idx_in_batch
            if global_idx < len(image_metadata):
                img_name, img_type = image_metadata[global_idx]
            else:
                img_name, img_type = f"image_{global_idx:06d}", "unknown"
            all_image_names.append(img_name)
            all_image_types.append(img_type)

            if collect_all:
                all_losses.append(batch_losses[idx_in_batch])
                all_wers.append(wer)
                all_cers.append(cer)
                # collect the already-stripped pred/gt (not the raw preds_str)
                # so the CSV gets clean strings and compute_cer_wer() in
                # evaluate.py sees the same strings that produced the metrics above
                all_preds.append(pred)
                all_labels.append(gt)
                all_confidence.append(confidence_score_list[-1] if confidence_score_list else 0.0)

        # all_preds/all_labels/all_confidence are now filled per-sample
        # inside the loop above when collect_all=True, so skip the old bulk extend

    # ── aggregate metrics ─────────────────────────────────────────────────────
    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)
    wer_avg = total_WER / float(length_of_data)
    cer_avg = total_CER / float(length_of_data)
    ser_avg = (total_SER / float(length_of_data)) * 100

    char_precision = char_TP / (char_TP + char_FP) if (char_TP + char_FP) > 0 else 0
    char_recall = char_TP / (char_TP + char_FN) if (char_TP + char_FN) > 0 else 0
    char_f1 = (
        2 * (char_precision * char_recall) / (char_precision + char_recall)
        if (char_precision + char_recall) > 0 else 0
    )

    avg_latency = (infer_time / length_of_data) * 1000 if length_of_data > 0 else 0

    ctc_loss_val = ctc_loss_avg.val() if 'CTC' in opt.Prediction else 0.0
    ce_loss_val = ce_loss_avg.val() if 'Attn' in opt.Prediction else 0.0

    result = (
        valid_loss_avg.val(),  # 0  overall validation loss
        accuracy,  # 1
        norm_ED,  # 2
        wer_avg,  # 3
        cer_avg,  # 4
        ser_avg,  # 5
        char_precision,  # 6
        char_recall,  # 7
        char_f1,  # 8
        avg_latency,  # 9
        preds_str,  # 10
        confidence_score_list,  # 11
        labels,  # 12
        infer_time,  # 13
        length_of_data,  # 14
        ctc_loss_val,  # 15  CTC loss
        ce_loss_val,  # 16  CE loss
    )
    if collect_all:
        return result + (all_preds, all_labels, all_confidence, all_image_names, all_image_types, all_losses, all_wers,
                         all_cers)
    return result