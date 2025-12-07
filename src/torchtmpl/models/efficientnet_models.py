# coding: utf-8

# Standard imports

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class GeM(nn.Module):
    """Generalized mean pooling layer with learnable parameter p."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.clamp(x, min=self.eps).pow(self.p)
        out = F.avg_pool2d(out, kernel_size=(out.shape[-2], out.shape[-1]))
        out = out.pow(1.0 / self.p)
        return out


def _adapt_stem_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
        dilation=conv.dilation,
        groups=conv.groups,
    )

    with torch.no_grad():
        weight = conv.weight
        if in_channels == 1:
            new_conv.weight.copy_(weight.mean(dim=1, keepdim=True))
        else:
            avg = weight.mean(dim=1, keepdim=True)
            repeats = (in_channels + avg.shape[1] - 1) // avg.shape[1]
            new_weight = avg.repeat(1, repeats, 1, 1)[:, :in_channels]
            new_conv.weight.copy_(new_weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class EfficientNetV2S(nn.Module):
    """EfficientNetV2-S backbone adapted for grayscale inputs and GeM pooling."""

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        use_pretrained = cfg.get("pretrained", False)
        weights = EfficientNet_V2_S_Weights.DEFAULT if use_pretrained else None
        backbone = efficientnet_v2_s(weights=weights)

        stem_conv = backbone.features[0][0]
        adapted_conv = _adapt_stem_conv(stem_conv, input_size[0])
        backbone.features[0][0] = adapted_conv

        self.features = backbone.features
        classifier = backbone.classifier
        dropout_module = classifier[0]
        self.dropout = nn.Dropout(p=dropout_module.p, inplace=False)
        self.feature_dim = classifier[1].in_features
        self.gem = GeM()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.features(x)
        x = self.gem(x).flatten(1)
        features = self.dropout(x)
        if return_features:
            return features
        return self.head(features)
