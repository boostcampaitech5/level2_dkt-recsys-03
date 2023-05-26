import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from omegaconf import DictConfig


def load_data(data_dir: str) -> pd.DataFrame:
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    train = pd.read_csv(path1)
    test = pd.read_csv(path2)
    test = get_valid(test)

    data = pd.concat([train, test])
    data.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    return data


def indexing_data(data: pd.DataFrame) -> Dict[str, int]:
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user = len(userid)
    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
    id2index = dict(userid2index, **itemid2index)
    return id2index


def process_data(data: pd.DataFrame, id2index: dict, config: DictConfig) -> dict:
    edge, label = [], []

    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id2index[user], id2index[item]
        edge.append([uid, iid])
        label.append(acode)
    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge, label=label)


def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    return train_data, test_data


def get_valid(data: pd.DataFrame) -> pd.DataFrame:
    data["idx"] = np.arange(0, data.shape[0])

    id_list = data.groupby("userID").nth(-2).idx
    data.loc[id_list, "answerCode"] = -2
    return data
