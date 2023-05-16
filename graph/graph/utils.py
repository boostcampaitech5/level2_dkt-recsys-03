import os
import torch
import random
import numpy as np
from datetime import datetime


def set_seeds(seed:int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_timestamp(date_format: str = '%d_%H%M%S') -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)