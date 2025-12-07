# coding: utf-8

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithLabelSmoothing(nn.Module):
    """Cross entropy loss with optional label smoothing."""

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        if self.smoothing <= 0:
            return F.nll_loss(log_probs, target)

        with torch.no_grad():
            smooth_value = self.smoothing / num_classes
            target_probs = torch.full_like(log_probs, smooth_value)
            target_probs.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + smooth_value)

        loss = (-target_probs * log_probs).sum(dim=1)
        return loss.mean()


class ArcMarginProduct(nn.Module):
    """ArcFace head with additive angular margin."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.unsqueeze(1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output