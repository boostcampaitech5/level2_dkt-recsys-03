import os
import wandb
import pandas as pd
import lightning as L
from datetime import datetime
from .model import LightGCNNet
from omegaconf import DictConfig
from .dataloader import GraphDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class Trainer:
    def __init__(self, dm: GraphDataModule, config: DictConfig):
        self.config = config
        self.sub_dm = GraphDataModule(config=self.config)
        self.val_dm = GraphDataModule(config=self.config, mode="val")
        self.model_name = self.config.model.model_name

        if self.model_name == "LightGCN":
            self.model = LightGCNNet(self.config)
            wandb.save(f"./configs/model/{self.model_name}.yaml")

    def train(self):
        now = datetime.now()
        file_name = now.strftime("%m%d_%H%M%S_") + self.config.model.model_name

        wandb_logger = WandbLogger()
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.config.trainer.patience, mode="min")
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=self.config.paths.output_path, filename=file_name + "{val_loss:.2f}")
        trainer = L.Trainer(
            max_epochs=self.config.trainer.epoch,
            log_every_n_steps=self.config.trainer.steps,
            logger=wandb_logger,
            callbacks=[early_stopping, checkpoint],
        )
        trainer.fit(self.model, datamodule=self.sub_dm)

        # save model to wandb
        wandb.save(checkpoint.best_model_path)

        if self.model_name == "LightGCN":
            final_model = LightGCNNet.load_from_checkpoint(checkpoint.best_model_path, config=self.config)
            final_model.eval()
            final_model.freeze()
        sub_predictions = trainer.predict(final_model, self.sub_dm)[0]
        val_predictions = trainer.predict(final_model, self.val_dm)[0]

        submission = pd.read_csv(self.config.paths.data_path + "sample_submission.csv")
        validation = pd.read_csv(self.config.paths.data_path + "validation.csv")

        # get predicted values from the list
        submission["prediction"] = sub_predictions
        validation["prediction"] = val_predictions

        sub_file_name = now.strftime("%m%d_%H%M%S_") + self.config.model.model_name + "_submit.csv"
        val_file_name = now.strftime("%m%d_%H%M%S_") + self.config.model.model_name + "_valid.csv"

        sub_write_path = os.path.join(self.config.paths.output_path, sub_file_name)
        val_write_path = os.path.join(self.config.paths.output_path, val_file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submission.to_csv(sub_write_path, index=False)
        validation.to_csv(val_write_path, index=False)

        print(f"Successfully saved submission as {sub_write_path}")
        print(f"Successfully saved submission as {val_write_path}")

        wandb.save(sub_write_path)
        wandb.save(val_write_path)
