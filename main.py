import torch
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


if __name__=="__main__":
    main()