#!/usr/bin/env python3
"""
Unit tests for Amharic CTC Label Converter and Tokenization
"""

import sys
import os
import unittest
import torch

# Add train_codes to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train_codes"))

from utils import CTCLabelConverter


class TestCTCLabelConverter(unittest.TestCase):

    def setUp(self):
        # Amharic sample vocabulary: base fidel, punctuation, numerals
        self.sample_alphabet = "ሀሁሂሃሄህሆለሉሊላሌልሎመሙሚማሜምሞ፩፪፫0123456789፡።፣፤ "
        self.converter = CTCLabelConverter(self.sample_alphabet)

    def test_blank_token_reservation(self):
        """Index 0 must be strictly reserved for [blank] token in CTC"""
        self.assertEqual(self.converter.character[0], "[blank]")
        self.assertNotIn("[blank]", self.converter.dict)
        self.assertGreater(len(self.converter.character), len(self.sample_alphabet))

    def test_encode_ethiopic_string(self):
        """Verify Amharic text encoding to integer tensors"""
        sample_texts = ["ሀሁሂ", "፩፪፫፡"]
        encoded_tensor, length_tensor = self.converter.encode(sample_texts)

        self.assertEqual(len(length_tensor), 2)
        self.assertEqual(length_tensor[0].item(), 3)
        self.assertEqual(length_tensor[1].item(), 4)
        self.assertEqual(len(encoded_tensor), 7)

    def test_greedy_decode_collapse_repeats(self):
        """Verify CTC greedy collapse of consecutive duplicate tokens and blanks"""
        # Encode "ሀ" and "ለ"
        h_idx = self.converter.dict["ሀ"]
        l_idx = self.converter.dict["ለ"]
        blank_idx = 0

        # Simulated raw frame sequence: [blank, ሀ, ሀ, blank, ለ, blank] -> should decode to "ሀለ"
        raw_seq = torch.IntTensor([blank_idx, h_idx, h_idx, blank_idx, l_idx, blank_idx])
        length = torch.IntTensor([len(raw_seq)])

        decoded_texts = self.converter.decode_greedy(raw_seq, length)
        self.assertEqual(len(decoded_texts), 1)
        self.assertEqual(decoded_texts[0], "ሀለ")

    def test_roundtrip_encode_decode(self):
        """Verify encoding and then decoding without duplicates returns original string"""
        text = "መሙሚ"
        encoded_tensor, length_tensor = self.converter.encode([text])
        decoded = self.converter.decode_greedy(encoded_tensor, length_tensor)
        self.assertEqual(decoded[0], text)


if __name__ == "__main__":
    unittest.main()
