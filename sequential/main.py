import os
import hydra
import torch
import wandb
import dotenv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from sequential.dataloader import DKTDataModule
from sequential.utils import set_seeds, get_logger, logging_conf, get_timestamp
from sequential.models import LSTM, LSTMATTN, BERT
from sequential.trainer import Trainer

logger = get_logger(logging_conf)

@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    # setting
    print(f"----------------- Setting -----------------")
    config.timestamp = get_timestamp()
    config.wandb.name = f'work-{get_timestamp()}'
    set_seeds(config.seed)

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)
    run.tags = [config.model.model_name, config.trainer.cv_strategy]
    wandb.save(f"./configs/config.yaml")

    # setup datamodule
    print(f"----------------- Setup datamodule -----------------")
    logger.info("Preparing data ...")
    dm = DKTDataModule(config)

    if config.trainer.cv_strategy == "holdout":
        trainer = Trainer(config, dm)
        trainer.train()
    else:
        raise NotImplementedError
    
    wandb.finish()


if __name__=="__main__":
    main()