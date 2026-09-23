import numpy as np
import pandas as pd
import os
import random
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import yaml

from utils import AttrDict, CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

cudnn.benchmark = True
cudnn.deterministic = False


def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(opt, show_number=2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data  = opt.select_data.split('-')
    opt.batch_ratio  = opt.batch_ratio.split('-')
    train_dataset    = Batch_Balanced_Dataset(opt)

    log = open(
        f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")

    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW,
        keep_ratio_with_pad=opt.PAD,
        contrast_adjust=opt.contrast_adjust
    )

    val_dataset, val_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(32, opt.batch_size),
        shuffle=True,
        num_workers=int(opt.workers),
        prefetch_factor=2,
        collate_fn=AlignCollate_valid,
        pin_memory=True
    )
    log.write(val_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters',
          opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel,
          opt.output_channel, opt.hidden_size, opt.num_class,
          opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        if opt.saved_model and os.path.exists(opt.saved_model):
            print(f"--- SUCCESS: Loading weights from {opt.saved_model} ---")
            pretrained_dict = torch.load(opt.saved_model, map_location='cpu')
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            raise FileNotFoundError(
                f"CRITICAL ERROR: Model not found at '{opt.saved_model}'. "
                f"Fix the path to avoid starting from scratch!"
            )

        if opt.new_prediction:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output,
                len(pretrained_dict['module.Prediction.weight'])
            )

        model = torch.nn.DataParallel(model).to(device)
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)            # # 1. Get the state dictionary of your current model setup
            # model_dict = model.state_dict()
            #
            # # 2. Filter out keys that don't match in shape
            # filtered_dict = {}
            # for k, v in pretrained_dict.items():
            #     if k in model_dict:
            #         if v.shape == model_dict[k].shape:
            #             filtered_dict[k] = v
            #         else:
            #             print(f"[Warning] Skipping weight loading for '{k}' due to shape mismatch: "
            #                   f"Checkpoint shape {v.shape} vs Current Model shape {model_dict[k].shape}")
            #     else:
            #         print(f"[Warning] Skipping weight loading for '{k}' because it doesn't exist in current model.")
            #
            # # 3. Load the safely filtered weights
            # model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)

        if opt.new_prediction:
            model.module.Prediction = nn.Linear(
                model.module.SequenceModeling_output, opt.num_class)
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
    else:
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)

    model.train()
    print("Model:")
    print(model)
    count_parameters(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    if opt.optim == 'adam':
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, weight_decay=float(opt.weight_decay), eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy       = -1
    best_norm_ED        = -1
    best_wer            = float('inf')
    best_cer            = float('inf')
    best_ser            = float('inf')
    best_char_precision = -1
    best_char_recall    = -1
    best_char_f1        = -1
    best_latency        = float('inf')
    best_ctc_loss       = float('inf')   # lower is better — CTC path
    best_ce_loss        = float('inf')   # lower is better — Attention path
    i = start_iter

    scaler = GradScaler()
    t1 = time.time()

    while True:
        # ── train step ────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(
                    labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device),
                                     preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])
                    target = text[:, 1:]
                    cost = criterion(
                        preds.view(-1, preds.shape[-1]),
                        target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(
                labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device),
                                 preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])
                target = text[:, 1:]
                cost = criterion(
                    preds.view(-1, preds.shape[-1]),
                    target.contiguous().view(-1))

            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        loss_avg.add(cost)

        # ── validation ────────────────────────────────────────────────────────
        if (i % opt.valInterval == 0) and (i != 0):
            print('training time: ', time.time() - t1)
            t1 = time.time()
            elapsed_time = time.time() - start_time

            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    # unpack all 17 return values
                    (valid_loss,
                     current_accuracy,
                     current_norm_ED,
                     current_wer,
                     current_cer,
                     current_ser,
                     current_char_precision,
                     current_char_recall,
                     current_char_f1,
                     current_latency,
                     preds,
                     confidence_score,
                     labels,
                     infer_time,
                     length_of_data,
                     current_ctc_loss,    # 15 — non-zero only for CTC
                     current_ce_loss,     # 16 — non-zero only for Attention
                    ) = validation(
                        model, criterion, val_loader, converter, opt, device)
                model.train()

                # ── loss header ───────────────────────────────────────────────
                loss_log = (
                    f'[{i}/{opt.num_iter}] '
                    f'Train loss: {loss_avg.val():0.5f}, '
                    f'Valid loss: {valid_loss:0.5f}, '
                    + (f'CTC loss: {current_ctc_loss:0.5f}, '
                       if 'CTC'  in opt.Prediction else '')
                    + (f'CE loss: {current_ce_loss:0.5f}, '
                       if 'Attn' in opt.Prediction else '')
                    + f'Elapsed_time: {elapsed_time:0.5f}'
                )
                loss_avg.reset()

                # ── per-metric current/best rows ──────────────────────────────
                current_model_log = (
                    f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, '
                    f'{"Current_norm_ED":17s}: {current_norm_ED:0.4f}, '
                    f'{"Current_WER":17s}: {current_wer:0.4f}, '
                    f'{"Current_CER":17s}: {current_cer:0.4f}, '
                    f'{"Current_SER":17s}: {current_ser:0.2f}%, '
                    f'{"Char_Precision":17s}: {current_char_precision:0.4f}, '
                    f'{"Char_Recall":17s}: {current_char_recall:0.4f}, '
                    f'{"Char_F1":17s}: {current_char_f1:0.4f}, '
                    f'{"Latency(ms)":17s}: {current_latency:0.2f}'
                    + (f', {"CTC_Loss":17s}: {current_ctc_loss:0.5f}'
                       if 'CTC'  in opt.Prediction else '')
                    + (f', {"CE_Loss":17s}: {current_ce_loss:0.5f}'
                       if 'Attn' in opt.Prediction else '')
                )

                # ── best-model checkpointing ──────────────────────────────────
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                if current_wer < best_wer:
                    best_wer = current_wer
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_wer.pth')
                if current_cer < best_cer:
                    best_cer = current_cer
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_cer.pth')
                if current_ser < best_ser:
                    best_ser = current_ser
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_ser.pth')
                if current_char_precision > best_char_precision:
                    best_char_precision = current_char_precision
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_char_precision.pth')
                if current_char_recall > best_char_recall:
                    best_char_recall = current_char_recall
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_char_recall.pth')
                if current_char_f1 > best_char_f1:
                    best_char_f1 = current_char_f1
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_char_f1.pth')
                if current_latency < best_latency:
                    best_latency = current_latency
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_latency.pth')

                # save checkpoint when validation CTC loss improves
                if 'CTC' in opt.Prediction and current_ctc_loss < best_ctc_loss:
                    best_ctc_loss = current_ctc_loss
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_ctc_loss.pth')

                # save checkpoint when validation CE loss improves (Attention path)
                if 'Attn' in opt.Prediction and current_ce_loss < best_ce_loss:
                    best_ce_loss = current_ce_loss
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.experiment_name}/best_ce_loss.pth')

                best_model_log = (
                    f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, '
                    f'{"Best_norm_ED":17s}: {best_norm_ED:0.4f}, '
                    f'{"Best_WER":17s}: {best_wer:0.4f}, '
                    f'{"Best_CER":17s}: {best_cer:0.4f}, '
                    f'{"Best_SER":17s}: {best_ser:0.2f}%, '
                    f'{"Best_Char_P":17s}: {best_char_precision:0.4f}, '
                    f'{"Best_Char_R":17s}: {best_char_recall:0.4f}, '
                    f'{"Best_Char_F1":17s}: {best_char_f1:0.4f}, '
                    f'{"Best_Latency":17s}: {best_latency:0.2f}ms'
                    + (f', {"Best_CTC_Loss":17s}: {best_ctc_loss:0.5f}'
                       if 'CTC'  in opt.Prediction else '')
                    + (f', {"Best_CE_Loss":17s}: {best_ce_loss:0.5f}'
                       if 'Attn' in opt.Prediction else '')
                )

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # ── sample predictions ────────────────────────────────────────
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

                start = random.randint(0, len(labels) - show_number)
                for gt, pred, confidence in zip(
                    labels[start:start + show_number],
                    preds[start:start + show_number],
                    confidence_score[start:start + show_number]
                ):
                    if 'Attn' in opt.Prediction:
                        gt   = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time() - t1)
                t1 = time.time()

        # save checkpoint every 1e+4 iterations
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(),
                f'./saved_models/{opt.experiment_name}/iter_{i + 1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'train.csv')
            df = pd.read_csv(
                csv_path, sep='^([^,]+),', engine='python',
                usecols=['img_filename', 'img_transcription'],
                keep_default_na=False)
            all_char = ''.join(df['img_transcription'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_config("config_files/amh_filtered_config.yaml")
    train(opt, amp=False)
