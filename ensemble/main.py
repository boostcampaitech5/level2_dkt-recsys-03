import os
import hydra
import numpy as np
from omegaconf import DictConfig

from ensemble.stacking import Stacking
from ensemble.utils import set_seeds


def __main(config: DictConfig = None) -> None:
    # set seed
    set_seeds(config.seed)

    # load ensemble startegy model
    model = Stacking(filenames=config.ensemble_list, filepath=config.ensemble_path, seed=config.seed, test_size=config.test_size)

    # train & inference
    model.train()
    weights = model.get_weights()
    result = model.infer()

    # make save dir
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # save inference result
    submit = model.submit_frame.copy()
    submit["prediction"] = result

    weights_info = "".join([str(w)[:4] for w in weights])
    file_title = "-".join(config.ensemble_list)

    csv_path = f"{config.output_path}{weights_info}-{file_title}.csv"
    submit.to_csv(csv_path, index=False)

    print(f"========= new output saved : {csv_path} =========")


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
