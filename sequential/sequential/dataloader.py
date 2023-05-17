import os
import time
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder


class DKTDataset(Dataset):
    def __init__(self, data: np.ndarray, config: DictConfig):
        self.data = data
        self.max_seq_len = config.data.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # Load from data
        test, question, tag, correct, priorSolvingTime, testType = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
            "prior_solving_time": torch.tensor(priorSolvingTime, dtype=torch.int),
            "test_type": torch.tensor(testType + 1, dtype=torch.int),
        }

        # gernerate mask & exec truncate or insert padding
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:  # truncate
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:  # pre-padding
            for k, seq in data.items():
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask.int()

        # generate interaction
        interaction = data["correct"] + 1  # plus 1 for padding
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


class DKTDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.pin_memory = False

    # Fill feature engineering func if needed using self.df, self.test_df
    def __feature_engineering(self):
        pass

    # Encode and Save/Load data
    def __preprocessing(self, is_train: bool = True):
        ##### testType 파생변수 생성
        if is_train:
            self.df["testType"] = self.df["assessmentItemID"].apply(lambda x: x[2]).astype(int)
        else:
            self.test_df["testType"] = (
                self.test_df["assessmentItemID"].apply(lambda x: x[2]).astype(int)
            )

        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "testType"]

        # convert time data to int timestamp
        def convert_time(s: str):
            timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
            return int(timestamp)

        if not os.path.exists(self.config.paths.asset_path):
            os.makedirs(self.config.paths.asset_path)

        for col in cate_cols:
            le = LabelEncoder()
            le_path = os.path.join(self.config.paths.asset_path, col + "_classes.npy")

            if is_train:
                a = self.df[col].unique().tolist() + ["unknown"]
                le.fit(a)  # encode str to int(0~N)
                np.save(le_path, le.classes_)  # save encoded data

                self.df[col] = self.df[col].astype(str)
                test = le.transform(self.df[col])
                self.df[col] = test
            else:
                le.classes_ = np.load(le_path)
                self.test_df[col] = self.test_df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

                self.test_df[col] = self.test_df[col].astype(str)
                test = le.transform(self.test_df[col])
                self.test_df[col] = test

        if is_train:
            self.df["Timestamp"] = self.df["Timestamp"].apply(convert_time)
        else:
            self.test_df["Timestamp"] = self.test_df["Timestamp"].apply(convert_time)

        ### get diff
        if is_train:
            selected = self.df[["Timestamp", "userID"]]
            diff = selected.groupby(["userID"]).diff()

            prior_solving_time = diff["Timestamp"].fillna(0).clip(0, 300)
            self.df["priorSolvingTime"] = prior_solving_time
        else:
            t_selected = self.test_df[["Timestamp", "userID"]]
            t_diff = t_selected.groupby(["userID"]).diff()

            t_prior_solving_time = t_diff["Timestamp"].fillna(0).clip(0, 300)
            self.test_df["priorSolvingTime"] = t_prior_solving_time

    # split train data to train & valid : this part will be excahnged
    def split_data(
        self, data: np.ndarray, ratio: float = 0.7, shuffle: bool = True, seed: int = 42
    ):
        seed = self.config.seed
        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        size = int(len(data) * ratio)
        self.train_data = data[:size]
        self.valid_data = data[size:]

    # load and feature_engineering dataset
    def prepare_data(self):
        train_file_path = os.path.join(self.config.paths.data_path, self.config.paths.train_file)
        test_file_path = os.path.join(self.config.paths.data_path, self.config.paths.test_file)

        self.df = pd.read_csv(train_file_path)
        self.test_df = pd.read_csv(test_file_path)
        self.__feature_engineering()  # fill this func if needed

    # preprocess and set dataset on train/test case
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.__preprocessing()
        if stage == "predict" or stage is None:
            self.__preprocessing(is_train=False)

        self.n_questions = len(
            np.load(os.path.join(self.config.paths.asset_path, "assessmentItemID_classes.npy"))
        )
        self.n_tests = len(
            np.load(os.path.join(self.config.paths.asset_path, "testId_classes.npy"))
        )
        self.n_tags = len(
            np.load(os.path.join(self.config.paths.asset_path, "KnowledgeTag_classes.npy"))
        )

        columns = [
            "userID",
            "assessmentItemID",
            "testId",
            "answerCode",
            "KnowledgeTag",
            "priorSolvingTime",
            "testType",
        ]

        if stage == "fit" or stage is None:
            self.df = self.df.sort_values(by=["userID", "Timestamp"], axis=0)
            group = (
                self.df[columns]
                .groupby("userID")
                .apply(
                    lambda r: (
                        r["testId"].values,
                        r["assessmentItemID"].values,
                        r["KnowledgeTag"].values,
                        r["answerCode"].values,
                        r["priorSolvingTime"].values,
                        r["testType"].values,
                    )
                )
            )
            self.split_data(group.values)

        if stage == "predict" or stage is None:
            self.test_df = self.test_df.sort_values(by=["userID", "Timestamp"], axis=0)
            group = (
                self.test_df[columns]
                .groupby("userID")
                .apply(
                    lambda r: (
                        r["testId"].values,
                        r["assessmentItemID"].values,
                        r["KnowledgeTag"].values,
                        r["answerCode"].values,
                        r["priorSolvingTime"].values,
                        r["testType"].values,
                    )
                )
            )
            self.test_data = group.values

    def train_dataloader(self):
        trainset = DKTDataset(self.train_data, self.config)
        return DataLoader(
            trainset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        valset = DKTDataset(self.valid_data, self.config)
        return DataLoader(
            valset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        testset = DKTDataset(self.test_data, self.config)
        return DataLoader(
            testset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.pin_memory,
        )


class DKTDataKFoldModule(DKTDataModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            pass
        if stage == "predict" or stage is None:
            pass
