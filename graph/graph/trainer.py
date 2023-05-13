import os
import pandas as pd
import lightning as L
from .model import LightGCNNet
from datetime import datetime
from .dataloader import GraphDataModule


class Trainer:
    def __init__(self, dm: GraphDataModule):
        self.dm = dm
        self.model = LightGCNNet()

    def train(self):
        trainer = L.Trainer(max_epochs=10, log_every_n_steps=1)
        trainer.fit(self.model,datamodule=self.dm)

        predictions = trainer.predict(self.model, datamodule=self.dm)
        submission = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
        submission['prediction'] = predictions[0]


        now = datetime.now()
        file_name = now.strftime('%m%d_%H%M%S_') + 'LightGCN' + "_submit.csv"

        write_path = os.path.join('outputs/', file_name)
        os.makedirs(name='outputs/', exist_ok=True)

        submission.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")