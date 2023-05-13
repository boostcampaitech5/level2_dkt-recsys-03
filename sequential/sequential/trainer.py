import os
import wandb
import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig

from .dataloader import DKTDataModule
from .models import LSTM, LSTMATTN, BERT
from .metrics import get_metric
from .utils import get_logger, logging_conf

logger = get_logger(logging_conf)

class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config

    def load_data(self):
        print(f"----------------- Setup datamodule -----------------")
        logger.info("Preparing data ...")
        return DKTDataModule(self.config)

    def load_model(self):
        logger.info("Building Model ...")
        if self.config.model.model_name == "LSTM":
            wandb.save(f"./configs/model/LSTM.yaml")
            return LSTM(self.config)
        elif self.config.model.model_name == "LSTMATTN":
            wandb.save(f"./configs/model/LSTMATTN.yaml")
            return LSTMATTN(self.config)
        elif self.config.model.model_name == "BERT":
            wandb.save(f"./configs/model/BERT.yaml")
            return BERT(self.config)
        else:
            raise Exception(f"Wrong model name is used : {self.config.model.model_name}")

    def train(self):
        wandb_logger = WandbLogger(name=self.config.wandb.name, project=self.config.wandb.project)
        self.trainer = pl.Trainer(max_epochs = self.config.trainer.epoch, logger=wandb_logger)

        self.dm = self.load_data()
        self.model = self.load_model()

        logger.info("Start Training ...")
        self.trainer.fit(self.model, datamodule=self.dm)
    
    def predict(self):
        logger.info("Making Prediction ...")
        predictions = self.trainer.predict(self.model, datamodule=self.dm)

        logger.info("Saving Submission ...")
        predictions = np.concatenate(predictions)
        submit_df = pd.DataFrame(predictions)
        submit_df = submit_df.reset_index()
        submit_df.columns = ['id', 'prediction']

        file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_submit.csv"

        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submit_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)


class KfoldTrainer(Trainer):
    def __init__(self, config: DictConfig, dm: DKTDataModule):
        super().__init__(config, dm)
        self.config = config
        self.dm = dm
    
    def cv(self):
        pass
    
    def oof(self):
        pass
