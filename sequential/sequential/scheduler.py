import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from omegaconf import DictConfig


def get_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig):
    if config.trainer.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="max", verbose=True)
    elif config.trainer.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.trainer.warmup_steps,
            num_training_steps=config.trainer.total_steps,
        )
    else:
        raise Exception(f"Wrong scheduler name is used : {config.trainer.scheduler}")
    return scheduler
