import torch
import torch.nn as nn
from torchvision import models


class PretrainedResNet34_FeatureExtractor(nn.Module):
    """
    Pretrained ResNet34 Feature Extractor modified for OCR/CRNN tasks.
    Modifies layer strides to preserve sequence width while aggressively reducing height.
    """

    def __init__(self, input_channel, output_channel=512):
        super(PretrainedResNet34_FeatureExtractor, self).__init__()

        # Load pretrained ResNet34
        try:
            from torchvision.models import ResNet34_Weights
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        except ImportError:
            resnet = models.resnet34(pretrained=True)

        # 1. Handle Input Channel Adaptation (Grayscale vs RGB)
        if input_channel != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(input_channel, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))

            # Initialize new weights by averaging the pretrained RGB channels
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
            resnet.conv1 = new_conv

        # 2. Modify Strides for OCR Text Recognition
        # Standard ResNet downsamples width too much. We change the stride of
        # layer3 and layer4 to (2, 1) to preserve horizontal character frequency.

        # Modify layer3 first block and its residual downsample connection
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)

        # Modify layer4 first block and its residual downsample connection
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        # 3. Strip away the classification head (avgpool and fc)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # 4. Collapse remaining height to 1 while leaving sequence width dynamic
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        # 5. Handle Output Channel Adaptation
        # ResNet34 natively outputs 512 channels. If a different output is requested,
        # apply a 1x1 convolution to resize the feature dimensions smoothly.
        if output_channel != 512:
            self.adapt_conv = nn.Conv2d(512, output_channel, kernel_size=1)
        else:
            self.adapt_conv = nn.Identity()

    def forward(self, input):
        x = self.features(input)
        x = self.pool(x)
        x = self.adapt_conv(x)
        return x