import os
import hydra
from omegaconf import DictConfig

from ensemble.stacking import Stacking


def __main(config: DictConfig = None) -> None:
    # set seed

    # load ensemble startegy
    model = Stacking(config.ensemble_list, config.ensemble_path)

    # make final predict saving directory
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
