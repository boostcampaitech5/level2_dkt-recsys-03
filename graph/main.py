import hydra
from omegaconf import DictConfig
from graph.trainer import Trainer
from graph.dataloader import GraphDataModule


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config:DictConfig) -> None:
    dm = GraphDataModule(config)
    trainer = Trainer(dm, config)
    trainer.train()

if __name__ == "__main__":
    main()