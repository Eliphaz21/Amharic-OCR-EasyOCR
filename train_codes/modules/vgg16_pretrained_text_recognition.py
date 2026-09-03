import torch
import torch.nn as nn
from torchvision import models

class PretrainedVGG_FeatureExtractor(nn.Module):
    """ 
    Pretrained VGG16 Feature Extractor modified for OCR/CRNN tasks.
    It adapts the max pooling layers to preserve sequence width while reducing height.
    """
    def __init__(self, input_channel, output_channel=512):
        super(PretrainedVGG_FeatureExtractor, self).__init__()

        # Load pretrained VGG16 with Batch Normalization
        # Note: Using modern 'weights' API, with fallback to 'pretrained=True' for older PyTorch versions
        try:
            from torchvision.models import VGG16_BN_Weights
            vgg = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        except ImportError:
            vgg = models.vgg16_bn(pretrained=True)

        features = vgg.features
        self.ConvNet = nn.Sequential()

        pool_count = 0
        # Iterate through the pretrained layers and copy them over
        for i, layer in enumerate(features):
            if isinstance(layer, nn.MaxPool2d):
                pool_count += 1
                # Standard OCR modification: 
                # Keep the first two pools as 2x2. Change subsequent pools to 2x1 
                # so the width (text sequence) is preserved, but height is reduced.
                if pool_count > 2:
                    self.ConvNet.add_module(f'pool_{pool_count}', nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                else:
                    self.ConvNet.add_module(f'pool_{pool_count}', nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.ConvNet.add_module(f'layer_{i}', layer)

        # Handle Input Channel Adaptation (Grayscale vs RGB)
        if input_channel != 3:
            first_conv = getattr(self.ConvNet, 'layer_0')
            # Create a new Conv2d layer with the user's input_channel
            new_first_conv = nn.Conv2d(input_channel, first_conv.out_channels,
                                       kernel_size=first_conv.kernel_size,
                                       stride=first_conv.stride,
                                       padding=first_conv.padding,
                                       bias=(first_conv.bias is not None))
            
            # Initialize new weights by averaging the pretrained RGB channels
            # This allows the pretrained network to understand grayscale input immediately
            with torch.no_grad():
                new_first_conv.weight[:] = torch.mean(first_conv.weight, dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_first_conv.bias[:] = first_conv.bias

            # Replace the first layer
            setattr(self.ConvNet, 'layer_0', new_first_conv)

        # Handle Output Channel Adaptation
        # VGG16 natively outputs 512 channels. If a different output is requested, 
        # append a 1x1 convolution to adjust the channel size safely.
        if output_channel != 512:
            self.ConvNet.add_module('final_adapt_conv', nn.Conv2d(512, output_channel, kernel_size=1))

    def forward(self, input):
        return self.ConvNet(input)