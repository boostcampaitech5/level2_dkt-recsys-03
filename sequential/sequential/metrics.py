from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def get_metric(targets: np.ndarray, preds: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets.cpu().detach().numpy(), y_score=preds.cpu().detach().numpy())
    acc = accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=np.where(preds.cpu().detach().numpy() >= 0.5, 1, 0))
    return auc, acc
