import os
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder


class DKTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.df = pd.DataFrame()
    
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

    # load and feature_engineering dataset
    def prepare_data(self):
        train_file_path = os.path.join(self.args.data_path, self.args.train_file)
        self.df = pd.read_csv(train_file_path)
        self.__feature_engineering()  # fill this func if needed

    # preprocess and set dataset on train/test case
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.df = self.__preprocessing(is_train=True)
        if stage == "test" or stage is None:
            self.df = self.__preprocessing(is_train=False)
        
        # self.args_nqueionts
            
    def train_dataloader():
        # return DataLoder()
        pass

    def val_dataloader():
        # return DataLoder()
        pass

    def test_dataloader():
        # return DataLoder()
        pass
