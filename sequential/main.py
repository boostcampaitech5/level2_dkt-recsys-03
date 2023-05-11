import os
import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from sequential.dataloader import DKTDataModule
from sequential.args import parse_args
from sequential.utils import set_seeds, get_logger, logging_conf
from sequential.models import LSTM, LSTMATTN, BERT

logger = get_logger(logging_conf)

@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None)->None:
    # setting
    print(f"----------------- Setting -----------------")
    args = parse_args()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup datamodule
    print(f"----------------- Setup datamodule -----------------")
    logger.info("Preparing data ...")
    dm = DKTDataModule(config)
    logger.info("Building Model ...")
    model = LSTM()

    if config.cv_strategy == "holdout":
        logger.info("Building Model ...")
        # model = LSTMNATT()
        # model = BERT()

        trainer = pl.Trainer(max_epochs = args.epoch)

        logger.info("Start Training ...")
        trainer.fit(model, datamodule=dm)

        logger.info("Making Prediction ...")
        predictions = trainer.predict(model, datamodule=dm)

        logger.info("Saving Submission ...")
        predictions = np.concatenate(predictions)
        submit_df = pd.DataFrame(predictions)
        submit_df = submit_df.reset_index()
        submit_df.columns = ['id', 'prediction']

        write_path = os.path.join(args.output_path, "submissioin.csv")
        os.makedirs(name=args.output_path, exist_ok=True)
        submit_df.to_csv(write_path, index=False)

        print(f"Successfully saved submission as {write_path}")
    else:
        raise NotImplementedError


if __name__=="__main__":
    main()