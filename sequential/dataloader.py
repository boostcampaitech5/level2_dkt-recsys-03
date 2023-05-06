import os
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder

class DKTDataset(Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=troch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # gernerate mask & exec truncate or insert padding
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:  # truncate
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:  # pre-padding
            for k, seq in data.items():
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
    
        # generate interaction
        interaction = data["correct"] + 1  # plus 1 for padding
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)

class DKTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.df = pd.DataFrame()

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.pin_memory = False
    
    # Fill feature engineering func if needed using self.df
    def __feature_engineering(self):
        pass

    # Encode and Save/Load data
    def __preprocessing(self, is_train: bool = True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_path):
            os.makedirs(self.args.asset_path)

        for col in cate_cols:
            le = LabelEncoder()

            if is_train:
                a = self.df[col].unique().tolist() + ["unknown"]
                le.fit(a)  # encode str to int(0~N)
                le_path = os.path.join(self.args.asset_path, col + "_classes.npy")
                np.save(le_path, le.classes_)  # save encoded data
            else:
                le_path = os.path.join(self.args.asset_path, col + "_classes.npy")
                le.classes_ = np.load(le_path)
                self.df[col] = self.df[col].apply(lambda x: x if str(x) in le.classes_ else "unknown")

            self.df[col] = self.df[col].astype(str)
            test = le.transform(self.df[col])
            self.df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
            return int(timestamp)

        self.df["Timestamp"] = self.df["Timestamp"].apply(convert_time)

    # split train data to train & valid : this part will be excahnged
    def split_data(self, data: np.ndarray, ratio: float = 0.7, shuffle: bool = True, seed: int = 42):

        seed = self.args.seed
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        
        size = int(len(data) * ratio)
        self.train_data = data[:size]
        self.valid_data = data[size:]

    # load and feature_engineering dataset
    def prepare_data(self):
        train_file_path = os.path.join(self.args.data_path, self.args.train_file)
        self.df = pd.read_csv(train_file_path)
        self.__feature_engineering()  # fill this func if needed

    # preprocess and set dataset on train/test case
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.__preprocessing()
        # if stage == "test" or stage is None:
        #     self.__preprocessing(is_train=False)
        
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_path, "assessmentItemID_classes.npy")))
        self.args.n_tests = len(np.load(os.path.join(self.args.asset_path, "testId_classes.npy")))
        self.args.n_tags = len(np.load(os.path.join(self.args.asset_path, "KnowledgeTag_classes.npy")))

        self.df = self.df.sort_values(by=["userID","Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = self.df[columns].groupby("userID").apply(
            lambda r: (
                r["testId"].values,
                r["assessmentItemID"].values,
                r["KnowledgeTag"].values,
                r["answerCode"].values,
            )
        )

        if stage == "fit" or stage is None:
            self.split_data(group.values)
        # if stage == "test" or stage is None:
        #     self.test_data = group.values

    def train_dataloader(self):
        trainset = DKTDataset(self.train_data, self.args)
        print("trainset: ", len(trainset))
        return DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_worker=self.args.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        valset = DKTDataset(self.valid_data, self.args)
        print("valset: ", len(valset))
        return DataLoader(valset, batch_size=self.args.batch_size, shuffle=False, num_worker=self.args.num_workers, pin_memory=self.pin_memory)

    # def test_dataloader(self):
    #     testset = DKTDataset(self.test_data, self.args)
    #     print("testset: ", len(testset))
    #     return DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_worker=self.args.num_workers, pin_memory=self.pin_memory)
