import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def rmse(pred, test):
    return mean_squared_error(pred, test) ** 0.5


def get_timestamp(date_format: str = "%d_%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)
