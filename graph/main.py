from graph.trainer import Trainer
from graph.dataloader import GraphDataModule


def main():
    dm = GraphDataModule()
    trainer = Trainer(dm)
    trainer.train()

if __name__ == "__main__":
    main()