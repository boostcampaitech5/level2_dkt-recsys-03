import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def get_metric(true, prob) -> Tuple[float]:
    auc = roc_auc_score(true, prob)
    acc = accuracy_score(true, np.where(prob >= 0.5, 1, 0))
    return auc, acc
