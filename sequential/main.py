import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from sequential.dataloader import DKTDataModule
from sequential.args import parse_args
from sequential.utils import set_seeds
from sequential.models import LSTM

def main():
    args = parse_args()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # logger.info("Preparing data ...")

    dm = DKTDataModule(args)
    lstm = LSTM()

    trainer = pl.Trainer(max_epochs = args.epoch)

    print(">>> Start Fit !!!")
    trainer.fit(lstm, datamodule=dm)

    print(">>> Start Predict !!!")
    predictions = trainer.predict(lstm, datamodule=dm)

    predictions = np.concatenate(predictions)
    submit_df = pd.DataFrame(predictions)
    submit_df = submit_df.reset_index()
    submit_df.columns = ['id', 'prediction']

    write_path = os.path.join(args.output_path, "submissioin.csv")
    os.makedirs(name=args.output_path, exist_ok=True)
    submit_df.to_csv(write_path, index=False)

    print(f"Successfully saved submission as {write_path}")


if __name__=="__main__":
    main()