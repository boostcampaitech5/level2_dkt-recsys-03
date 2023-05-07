import os
import random
import numpy as np
from datetime import datetime


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_timestamp(date_format: str = '%d_%H%M%S') -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)
