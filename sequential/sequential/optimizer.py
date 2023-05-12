import torch
from torch.optim import Adam, AdamW
from omegaconf import DictConfig


def get_optimizer(param, config: DictConfig):
    optimizer = None
    if config.trainer.optimizer == "adam":
        optimizer = Adam(param, lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)
    elif config.trainer.optimizer == "adamW":
        optimizer = AdamW(param, lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)
    return optimizer
