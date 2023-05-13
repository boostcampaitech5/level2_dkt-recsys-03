import os
import pandas as pd
import lightning as L
from datetime import datetime
from .model import LightGCNNet
from omegaconf import DictConfig
from .dataloader import GraphDataModule


class Trainer:
    def __init__(self, dm: GraphDataModule, config: DictConfig):
        self.dm = dm
        self.config = config

        if self.config.model.model_name == "LightGCN":
            self.model = LightGCNNet(config)

    def train(self):
        trainer = L.Trainer(max_epochs=self.config.trainer.epoch, 
                            log_every_n_steps=self.config.trainer.steps)
        trainer.fit(self.model,datamodule=self.dm)

        predictions = trainer.predict(self.model, datamodule=self.dm)
        submission = pd.read_csv(self.config.paths.data_path + "sample_submission.csv")

        # get predicted values from the list
        submission['prediction'] = predictions[0]

        now = datetime.now()
        file_name = now.strftime('%m%d_%H%M%S_') + self.config.model.model_name + "_submit.csv"

        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submission.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
