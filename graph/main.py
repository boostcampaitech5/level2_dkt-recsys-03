import lightning as L
import pandas as pd
import os
from model import LightGCNNet
from datetime import datetime
from dataloader import GraphDataModule




def main():
    datamodule = GraphDataModule()
    model = LightGCNNet()
    trainer = L.Trainer(max_epochs=10, log_every_n_steps=1)
    trainer.fit(model,datamodule=datamodule)

    result = trainer.predict(model,datamodule=datamodule)
    submission = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
    submission['prediction'] = result[0]

    now = datetime.now()
    file_name = now.strftime('%m%d_%H%M%S_') + 'LightGCN' + "_submit.csv"

    write_path = os.path.join('outputs/', file_name)
    os.makedirs(name='outputs/', exist_ok=True)

    submission.to_csv(write_path, index=False)
    print(f"Successfully saved submission as {write_path}")


if __name__ == "__main__":
    main()
