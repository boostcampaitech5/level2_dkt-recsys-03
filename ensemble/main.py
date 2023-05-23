import os
import hydra
import wandb
import dotenv
import numpy as np
from omegaconf import DictConfig

from ensemble.stacking import Stacking, OofStacking
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

    wandb.log({"cv_strategy": config.cv_strategy})
    if config.cv_strategy == "holdout":
        model = Stacking(filenames=config.ensemble_list, filepath=config.ensemble_path, seed=config.seed, test_size=config.test_size)
    else:  # kfold
        model = OofStacking(filenames=config.ensemble_list, filepath=config.ensemble_path, seed=config.seed, test_size=config.test_size, k=config.k)

    # train & inference
    model.fit()
    submit_pred = model.infer()

    # make save dir
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # save inference result (submit_pred)
    submit = model.submit_frame.copy()
    submit["prediction"] = submit_pred

    csv_path = f"{config.output_path}{model.set_filename()}.csv"
    submit.to_csv(csv_path, index=False)
    wandb.save(csv_path)

    print(f"========= new output saved : {csv_path} =========")


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
