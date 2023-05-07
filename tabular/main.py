import os
import hydra
import omegaconf
from tabular.utils import seed_everything, get_timestamp
from tabular.data import TabularDataModule
from tabular.trainer import Trainer, CrossValidationTrainer


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: omegaconf.DictConfig=None) -> None:
    # setting
    print("Setting...")
    config.timestamp = get_timestamp()
    seed_everything(config.seed)
    print(f"timestamp: {config.timestamp}, seed: {config.seed}")
    # setup datamodule
    print("Setup datamodule...")
    datamodule = TabularDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()

    if config.cv_strategy == 'holdout':
        # trainer
        trainer = Trainer(config, datamodule)
        # train
        print("Training...")
        trainer.train()
        # inference
        print("Inference...")
        trainer.inference()
    
    elif config.cv_strategy == 'kfold':
        # trainer
        trainer = CrossValidationTrainer(config, datamodule)
        # train
        print("Training...")
        trainer.cv()
        # inference
        print("Inference...")
        trainer.oof()
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
