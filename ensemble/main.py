import os
import hydra
import wandb
import dotenv
import numpy as np
from omegaconf import DictConfig

from ensemble.ensembles import Ensemble
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

    if config.ensemble in ["weighted", "average"]:
        en = Ensemble(filenames=config.ensemble_list, filepath=config.ensemble_path)
        submit = en.submit_frame.copy()

        if config.ensemble == "weighted":
            submit_pred = en.simple_weighted(config.weight)
            csv_path = f"{config.output_path}sw-{en.set_filename()}.csv"
        else:  # average
            submit_pred = en.average_weighted()
            csv_path = f"{config.output_path}aw-{en.set_filename()}.csv"

    elif config.ensemble == "stacking":
        wandb.log({"cv_strategy": config.cv_strategy})

        if config.cv_strategy == "holdout":
            model = Stacking(
                filenames=config.ensemble_list,
                filepath=config.ensemble_path,
                seed=config.seed,
                test_size=config.test_size,
                intercept_opt=config.intercept_opt,
            )
        else:  # kfold
            model = OofStacking(
                filenames=config.ensemble_list,
                filepath=config.ensemble_path,
                seed=config.seed,
                test_size=config.test_size,
                k=config.k,
                intercept_opt=config.intercept_opt,
            )

        # train & inference
        model.fit()
        submit_pred = model.infer()
        submit = model.submit_frame.copy()

        if config.intercept_opt:
            csv_path = f"{config.output_path}stk-{model.set_filename()}.csv"
        else:
            csv_path = f"{config.output_path}stk-nointer-{model.set_filename()}.csv"

    # make save dir
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # save inference result (submit_pred)
    submit["prediction"] = submit_pred
    submit["prediction"] = submit.apply(lambda x: 1 if x["prediction"] > 1 else (0 if x["prediction"] < 0 else x["prediction"]), axis=1)

    submit.to_csv(csv_path, index=False)
    wandb.save(csv_path)

    print(f"========= new output saved : {csv_path} =========")


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
