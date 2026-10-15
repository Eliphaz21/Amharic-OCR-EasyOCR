#!/usr/bin/env python3
"""
Unit tests for Amharic OCR CRNN Model Architecture & Forward Pass
"""

import sys
import os
import unittest
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train_codes"))

try:
    import torchvision
    HAS_TORCHVISION = True
    from model import Model
    from utils import AttrDict
except ImportError:
    HAS_TORCHVISION = False


@unittest.skipUnless(HAS_TORCHVISION, "torchvision required for model forward pass test")
class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.opt = AttrDict({
            "Transformation": "None",
            "FeatureExtraction": "VGG",
            "SequenceModeling": "BiLSTM",
            "Prediction": "CTC",
            "num_fiducial": 20,
            "imgH": 64,
            "imgW": 300,
            "input_channel": 1,
            "output_channel": 512,
            "hidden_size": 256,
            "num_class": 435,
            "batch_max_length": 100,
        })
        self.model = Model(self.opt)
        self.model.eval()

    def test_forward_pass_output_shape(self):
        """Verify model output shape on batch of size 2 with 64x300 images"""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 64, 300)
        dummy_text = None

        with torch.no_grad():
            preds = self.model(dummy_input, dummy_text, is_train=False)

        # Output shape should be [batch_size, time_steps, num_class]
        self.assertEqual(preds.size(0), batch_size)
        self.assertEqual(preds.size(2), self.opt.num_class)
        self.assertGreater(preds.size(1), 0)

    def test_parameter_count(self):
        """Verify model has expected trainable parameter scale (>10M)"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(trainable_params, 10_000_000)


if __name__ == "__main__":
    unittest.main()
