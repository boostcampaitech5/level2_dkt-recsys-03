import os
import hydra
import omegaconf
from tabular.utils import seed_everything, get_timestamp
from tabular.data import TabularDataModule
from tabular.trainer import Trainer


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: omegaconf.DictConfig=None) -> None:
    # setting
    print("Setting...")
    config.timestamp = get_timestamp()
    seed_everything(config.seed)
    # setup datamodule
    print("Setup datamodule...")
    datamodule = TabularDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()
    # trainer
    trainer = Trainer(config, datamodule)
    trainer.init_model()
    # train
    print("train...")
    trainer.train()
    # inference
    print("inference...")
    trainer.inference()
    

if __name__ == "__main__":
    main()
