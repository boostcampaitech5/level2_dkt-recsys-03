import os
import hydra
import wandb
import dotenv
import numpy as np
from omegaconf import DictConfig

from ensemble.stacking import Stacking
from ensemble.utils import set_seeds, get_timestamp


def __main(config: DictConfig = None) -> None:
    # setting
    print(f"----------------- Setting -----------------")
    set_seeds(config.seed)
    config.wandb_name = f"ens-{get_timestamp()}"

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
    )

    run.tags = config.ensemble_list
    wandb.save("./configs/config.yaml")

    # load ensemble startegy model
    print(f"Models: {config.ensemble_list}")
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
    wandb.save(csv_path)

    print(f"========= new output saved : {csv_path} =========")


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
