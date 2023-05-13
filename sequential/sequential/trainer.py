import os
import wandb
import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from omegaconf import DictConfig

from .dataloader import DKTDataModule
from .models import LSTM, LSTMATTN, BERT
from .metrics import get_metric
from .utils import get_logger, logging_conf

logger = get_logger(logging_conf)

class Trainer:
    def __init__(self, config: DictConfig, dm: DKTDataModule):
        self.config = config
        self.dm = dm

        # set model
        logger.info("Building Model ...")
        if self.config.model.model_name == "LSTM":
            self.model = LSTM(config)
            wandb.save(f"./configs/model/LSTM.yaml")
        elif self.config.model.model_name == "LSTMATTN":
            self.model = LSTMATTN(config)
            wandb.save(f"./configs/model/LSTMATTN.yaml")
        elif self.config.model.model_name == "BERT":
            self.model = BERT(config)
            wandb.save(f"./configs/model/BERT.yaml")
        else:
            raise Exception(f"Wrong model name is used : {self.config.model.model_name}")

    def train(self):
        wandb_logger = WandbLogger(name=self.config.wandb.name, project=self.config.wandb.project)
        trainer = pl.Trainer(max_epochs = self.config.trainer.epoch, logger=wandb_logger)

        logger.info("Start Training ...")
        trainer.fit(self.model, datamodule=self.dm)

        logger.info("Making Prediction ...")
        predictions = trainer.predict(self.model, datamodule=self.dm)

        logger.info("Saving Submission ...")
        predictions = np.concatenate(predictions)
        submit_df = pd.DataFrame(predictions)
        submit_df = submit_df.reset_index()
        submit_df.columns = ['id', 'prediction']

        now = datetime.now()
        file_name = now.strftime('%m%d_%H%M%S_') + self.config.model.model_name + "_submit.csv"

        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submit_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)