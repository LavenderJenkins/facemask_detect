import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv3(nn.Module):
    """
    Lightweight multi-scale YOLOv3-like module.

    - Provides a small backbone that downsamples the input and produces
      three feature maps at different scales.
    - Returns a list of 3 output tensors (one per scale). Each tensor has
      shape (B, anchors*(5+num_classes), H, W).

    This keeps the API compatible with training code that does `outputs = model(imgs)`.
    """

    def __init__(self, num_classes=3, img_size=416, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors_per_scale = anchors_per_scale

        out_ch = (self.num_classes + 5) * self.anchors_per_scale

        # Small backbone that progressively downsamples the input
        # Input  ->  HxW
        # layer1 -> 1/2
        # layer2 -> 1/4
        # layer3 -> 1/8
        # layer4 -> 1/16  (we'll use 3 scales from layer4, layer3, layer2)
        self.layer1 = nn.Sequential(
            ConvBNLeaky(3, 32, 3, 1, 1),
            ConvBNLeaky(32, 64, 3, 2, 1),
            ConvBNLeaky(64, 64, 3, 1, 1),
        )

        self.layer2 = nn.Sequential(
            ConvBNLeaky(64, 128, 3, 2, 1),
            ConvBNLeaky(128, 128, 3, 1, 1),
        )

        self.layer3 = nn.Sequential(
            ConvBNLeaky(128, 256, 3, 2, 1),
            ConvBNLeaky(256, 256, 3, 1, 1),
        )

        self.layer4 = nn.Sequential(
            ConvBNLeaky(256, 512, 3, 2, 1),
            ConvBNLeaky(512, 512, 3, 1, 1),
        )

        # Detection heads for three scales (1x1 conv to output desired channel count)
        self.head_small = nn.Conv2d(512, out_ch, kernel_size=1)
        self.head_medium = nn.Conv2d(256, out_ch, kernel_size=1)
        self.head_large = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.layer1(x)   # downsampled by 2
        f_large = self.layer2(x)   # downsampled by 4
        f_medium = self.layer3(f_large)  # downsampled by 8
        f_small = self.layer4(f_medium)  # downsampled by 16

        out_small = self.head_small(f_small)
        out_medium = self.head_medium(f_medium)
        out_large = self.head_large(f_large)

        # Return list of outputs (same convention used by many YOLO implementations)
        return [out_small, out_medium, out_large]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["YOLOv3", "count_parameters"]
