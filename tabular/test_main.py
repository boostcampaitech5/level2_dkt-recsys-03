from unittest.mock import patch, MagicMock
from main import main


@patch('wandb.init')
@patch('wandb.login')
@patch('wandb.log')
@patch('wandb.finish')
@patch('wandb.save')
@patch('wandb.lightgbm')
@patch('wandb.run', new=MagicMock())
@patch('wandb.plot.confusion_matrix', new=MagicMock())
def test_main(*args, **kwargs):
    main()