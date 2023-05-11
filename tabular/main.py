import os
import hydra
import omegaconf
import dotenv
import wandb
from tabular.utils import seed_everything, get_timestamp
from tabular.data import TabularDataModule
from tabular.trainer import Trainer, CrossValidationTrainer


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: omegaconf.DictConfig=None) -> None:
    # setting
    print(f'--------------- Setting ---------------')
    config.timestamp = get_timestamp()
    config.wandb.name = f'work-{get_timestamp()}'
    seed_everything(config.seed)
    
    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)
    run.tags = [config.model.name, config.cv_strategy]
    wandb.save(f"./configs/config.yaml")
    wandb.save(f"./configs/model/LGBM.yaml")
    
    # setup datamodule
    print(f'--------------- Setup datamodule ---------------')
    datamodule = TabularDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()

    if config.cv_strategy == 'holdout':
        # trainer
        trainer = Trainer(config, datamodule)
        # train
        print(f'--------------- Training ---------------')
        trainer.train()
        # inference
        print(f'--------------- Inference ---------------')
        trainer.inference()
    
    elif config.cv_strategy == 'kfold':
        # trainer
        trainer = CrossValidationTrainer(config, datamodule)
        # train
        print(f'--------------- Training ---------------')
        trainer.cv()
        # inference
        print(f'--------------- Inference ---------------')
        trainer.oof()
    
    else:
        raise NotImplementedError
    
    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()