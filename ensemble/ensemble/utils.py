import os
from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_metric(targets: np.ndarray, preds: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets, y_score=preds)
    acc = accuracy_score(y_true=targets, y_pred=np.where(preds >= 0.5, 1, 0))
    return auc, acc


def get_timestamp(date_format: str = "%d_%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)
