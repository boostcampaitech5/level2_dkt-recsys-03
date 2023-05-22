import os
import hydra
import omegaconf
import dotenv
import wandb
from tabular.utils import seed_everything, get_timestamp
from tabular.data import TabularDataModule
from tabular.trainer import Trainer, CrossValidationTrainer


def __main(config: omegaconf.DictConfig = None) -> None:
    # turn to absolute path
    config.paths.data_dir = os.path.expanduser(config.paths.data_dir)

    # setting
    print("--------------- Setting ---------------")
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{config.timestamp}"
    seed_everything(config.seed)

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)
    run.tags = [config.model.name, config.cv_strategy]
    wandb.save("./configs/config.yaml")
    wandb.save("./configs/model/LGBM.yaml")

    # setup datamodule
    print("--------------- Setup datamodule ---------------")
    datamodule = TabularDataModule(config)
    # process data
    if config.skip_data_processing:
        print("Skip data processing, just load data files")
        datamodule.shortcut()
    else:
        print("Load data files, and process data")
        datamodule.prepare_data()
        datamodule.setup()
    wandb.run.summary["data_version"] = config.data_version

    if config.cv_strategy == "holdout":
        # trainer
        trainer = Trainer(config, datamodule)
        # train
        print("--------------- Training ---------------")
        trainer.train()
        # inference
        print("--------------- Inference ---------------")
        trainer.inference()

    elif config.cv_strategy == "kfold":
        # trainer
        trainer = CrossValidationTrainer(config, datamodule)
        # train
        print("--------------- Training ---------------")
        trainer.cv()
        # inference
        print("--------------- Inference ---------------")
        trainer.oof()

    else:
        raise NotImplementedError

    # wandb finish
    wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: omegaconf.DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
