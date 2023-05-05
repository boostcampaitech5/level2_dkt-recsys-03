import torch
import pytorch_lightning as pl
from sequential.dataloader import DKTDataModule
from sequential.args import parse_args
from sequential.utils import set_seeds

def main():
    args = parse_args()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = DKTDataModule(args)
    # 모델 = modelclass()

    trainer = pl.Trainer(max_epochs = args.epoch)
    # trainer.fit(모델, datamodule=dm)

if __name__=="__main__":
    main()