import os
import hydra
import omegaconf
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock
from main import __main


@patch("wandb.init")
@patch("wandb.login")
@patch("wandb.log")
@patch("wandb.finish")
@patch("wandb.save")
@patch("wandb.lightgbm")
@patch("wandb.run", new=MagicMock())
@patch("wandb.plot.confusion_matrix", new=MagicMock())
def test_main(*args, **kwargs):
    @hydra.main(version_base="1.2", config_path="configs", config_name="test.yaml")
    def inner_main(config: DictConfig) -> None:
        for model_path in os.listdir("configs/model.test"):
            model_path = os.path.join("configs/model.test", model_path)

            model_config = omegaconf.OmegaConf.load(model_path)
            config.model = model_config

            __main(config)

    inner_main()
