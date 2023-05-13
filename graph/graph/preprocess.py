import os
import pandas as pd
from typing import Tuple, Dict
import torch


def prepare_dataset( data_dir: str) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir)
    train_data, test_data = separate_data(data=data)
    id2index: dict = indexing_data(data=data)
    train_data_proc = process_data(data=train_data, id2index=id2index)
    test_data_proc = process_data(data=test_data, id2index=id2index)
    return train_data_proc, test_data_proc, len(id2index)

def load_data(data_dir: str) -> pd.DataFrame: 
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data = pd.concat([data1, data2])
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

def process_data(data: pd.DataFrame, id2index: dict) -> dict:
    edge, label = [], []

    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id2index[user], id2index[item]
        edge.append([uid, iid])
        label.append(acode)
        
    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge,
                label=label)

def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    return train_data, test_data