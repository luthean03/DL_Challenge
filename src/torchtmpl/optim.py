# coding: utf-8

from .losses import CrossEntropyWithLabelSmoothing


def get_loss(cfg):
    name = cfg["name"]
    smoothing = cfg.get("label_smoothing", 0.0)
    if name == "CrossEntropy":
        return CrossEntropyWithLabelSmoothing(smoothing)
    raise ValueError(f"Unknown loss name: {name}")


def get_optimizer(cfg, params):
    algo = cfg["algo"]
    params_dict = cfg.get("params", {})
    optim_cls = getattr(__import__("torch.optim", fromlist=[algo]), algo)
    return optim_cls(params, **params_dict)