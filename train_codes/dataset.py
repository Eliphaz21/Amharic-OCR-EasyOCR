import os
import sys
import re
import six
import math
import torch
import pandas as pd

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from itertools import accumulate as _accumulate
import torchvision.transforms as transforms
import albumentations as A  # <-- Added for advanced augmentations


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    # Add 1e-9 to prevent ZeroDivisionError
    return (high - low) / (high + low + 1e-9), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)

    # Check if high and low are the same (solid color image)
    # and if the contrast actually needs adjusting
    if contrast < target and high > low:
        img = img.astype(float)  # Use float for precise division

        # We add a safety check here
        diff = high - low
        ratio = 200. / diff

        img = (img - low + 25) * ratio

        # Clip values to stay between 0 and 255
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(
            f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 150
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        # Pass opt down to AlignCollate so it can determine if it should apply augmentations
        _AlignCollate = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(
                root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset *
                                 float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset,
                             total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(_dataset, batch_size=_batch_size, shuffle=True,
                                                       # prefetch_factor=2,persistent_workers=True,
                                                       num_workers=int(
                                                           opt.workers),
                                                       collate_fn=_AlignCollate, pin_memory=True)

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class OCRDataset(Dataset):

    def __init__(self, root, opt):
        self.root = root
        self.opt = opt

        absolute_root = os.path.abspath(self.root)
        parent_dir = os.path.dirname(absolute_root)

        # 1. Grab the name of the current folder ('train', 'val', or 'test')
        folder_name = os.path.basename(absolute_root)

        # 2. Dynamically build the CSV name: 'train_data.csv' or 'val_data.csv'
        csv_filename = f"{folder_name}.csv"
        label_path = os.path.join(parent_dir, csv_filename)

        print(f"--- [DEBUG] Loading {folder_name} labels from: {label_path} ---")

        self.df = pd.read_csv(label_path, sep=',', engine='python',
                              usecols=['img_filename', 'img_transcription'], keep_default_na=False)

        self.nSamples = len(self.df)

        if self.opt.data_filtering_off:
            self.filtered_index_list = [
                index for index in range(self.nSamples)]
        else:
            self.filtered_index_list = []
            for index in range(self.nSamples):
                # 3. FIXED: Using 'img_transcription' for filtering
                label = str(self.df.at[index, 'img_transcription'])
                try:
                    if len(label) > self.opt.batch_max_length:
                        continue
                except:
                    continue

                # Check for characters allowed in your Amharic alphabet
                out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):
                    continue
                self.filtered_index_list.append(index)

    def __len__(self):
        return len(self.filtered_index_list)

    def __getitem__(self, index):
        index = self.filtered_index_list[index]

        # 4. FIXED: Using 'img_filename' to find the file
        img_fname = self.df.at[index, 'img_filename']
        img_fpath = os.path.join(self.root, img_fname.strip())

        # 5. FIXED: Using 'img_transcription' as the text label
        label = str(self.df.at[index, 'img_transcription'])

        # Check if we should use RGB or Grayscale ('L')
        mode = 'RGB' if self.opt.rgb else 'L'

        try:
            img = Image.open(img_fpath).convert(mode)
        except FileNotFoundError:
            print(f"⚠️ Skipping missing file: {img_fpath}")
            img = Image.new(mode, (self.opt.imgW, self.opt.imgH), color=0)
        except Exception as e:
            print(f"❌ Error loading {img_fpath}: {e}")
            img = Image.new(mode, (self.opt.imgW, self.opt.imgH), color=0)

        if not self.opt.sensitive:
            label = label.lower()

        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)

        return (img, label)


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w -
                                          1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust=0., opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust
        self.opt = opt

        # Updated pipeline leveraging modern Albumentations range syntax to prevent server crashes
        self.augment_pipeline = A.Compose([
            # 1. ANTI-MEMORIZATION
            A.CoarseDropout(
                num_holes_range=(5, 25),
                hole_height_range=(2, 3),
                hole_width_range=(2, 3),
                fill=0,  # <-- Changed 'fill_value' to 'fill'
                p=0.5
            ),

            # 2. GEOMETRIC ROBUSTNESS
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.1, interpolation=1, p=1.0),
                A.ElasticTransform(alpha=1, sigma=20, p=1.0),  # <-- Removed 'alpha_affine' entirely
                A.Affine(rotate=(-2, 2), shear=(-2, 2), p=1.0),
            ], p=0.4),

            # 3. DEGRADATION SIMULATION
            A.OneOf([
                A.Morphological(scale=1, operation='erosion', p=1.0),
                A.ImageCompression(quality_range=(70, 95), p=1.0),  # <-- Combined into 'quality_range'
            ], p=0.2),

            # 4. LIGHTING VARIATION
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            )
        ])

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                # --- NEW AUGMENTATION ENGINE INJECTION ---
                # Only apply these structural distortions if we are currently training.
                # It evaluates whether 'is_train' exists in your options and if it is active.
                if self.opt and getattr(self.opt, 'is_train', True):
                    # Convert PIL image to NumPy array for Albumentations processing
                    img_np = np.array(image)
                    augmented = self.augment_pipeline(image=img_np)
                    img_np = augmented['image']
                    # Restore back into a clean PIL container
                    image = Image.fromarray(img_np, mode=images[0].mode)

                w, h = image.size

                # Pre-existing contrast adjustment step
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(
                        image, target=self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize(
                    (resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0)
                                       for t in resized_images], 0)

        else:
            # Fallback resizing branch if padding is disabled
            resized_images = []
            for image in images:
                if self.opt and getattr(self.opt, 'is_train', True):
                    img_np = np.array(image)
                    augmented = self.augment_pipeline(image=img_np)
                    image = Image.fromarray(augmented['image'], mode=images[0].mode)
                resized_images.append(image)

            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(img) for img in resized_images]
            image_tensors = torch.cat([t.unsqueeze(0)
                                       for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)