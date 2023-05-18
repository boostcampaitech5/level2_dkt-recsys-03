import os
import hydra
import wandb
import dotenv
from omegaconf import DictConfig
from graph.trainer import Trainer
from graph.utils import set_seeds, get_timestamp
from graph.dataloader import GraphDataModule


def __main(config: DictConfig) -> None:
    # setting
    print("+++++++setting++++++++")
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{get_timestamp()}"
    set_seeds(config.seed)

    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(entity=config.wandb.entity, project=config.wandb.project, name=config.wandb.name)
    run.tags = [config.model.model_name]
    wandb.save(f"./configs/config.yaml")

    dm = GraphDataModule(config)
    trainer = Trainer(dm, config)
    trainer.train()

    wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig) -> None:
    __main(config)


if __name__ == "__main__":
    main()
