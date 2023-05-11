import os
import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from sequential.dataloader import DKTDataModule
from sequential.utils import set_seeds, get_logger, logging_conf
from sequential.models import LSTM, LSTMATTN, BERT
from sequential.trainer import Trainer

logger = get_logger(logging_conf)

@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    # setting
    print(f"----------------- Setting -----------------")
    set_seeds(config.seed)

    # setup datamodule
    print(f"----------------- Setup datamodule -----------------")
    logger.info("Preparing data ...")
    dm = DKTDataModule(config)

    if config.cv_strategy == "holdout":
        trainer = Trainer(config, dm)
        trainer.train()
    else:
        raise NotImplementedError


if __name__=="__main__":
    main()