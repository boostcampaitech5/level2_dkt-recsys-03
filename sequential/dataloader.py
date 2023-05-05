import os
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder

class DKTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # define preprocess func as variable at here

    # load or download dataset
    def prepare_data(self):
        pass

    # define dataset and preprocess on train/val or test case
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            pass 
        if stage == "test" or stage is None:
            pass
    

    # define dataloader of each case
    def train_dataloader():
        # return DataLoder()
        pass

    def val_dataloader():
        # return DataLoder()
        pass

    def test_dataloader():
        # return DataLoder()
        pass
