import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
    # set random seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger():
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # set logging level

    # set logging foramt
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # set print log
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
