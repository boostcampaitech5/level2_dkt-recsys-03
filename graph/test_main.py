from unittest.mock import patch, MagicMock
from main import main


@patch('wandb.init')
@patch('wandb.login')
@patch('wandb.log')
@patch('wandb.finish')
@patch('wandb.save')
@patch("pytorch_lightning.loggers.wandb.Run", new=MagicMock)
@patch("pytorch_lightning.loggers.wandb.wandb")
def test_main(*args, **kwargs):
    main()