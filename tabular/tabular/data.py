import os
import random
import pandas as pd
from typing import Dict, List, Union, Optional
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.train_data_path: str = os.path.join(config.paths.data_dir, 'train_data.csv')
        self.test_data_path: str = os.path.join(config.paths.data_dir, 'test_data.csv')
        self.cv_strategy: str = config.cv_strategy

        self.train_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.valid_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self.train_dataset: Optional[TabularDataset] = None
        self.valid_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None

    def prepare_data(self):
        # load csv file
        train_data: pd.DataFrame = self.load_csv_file(self.train_data_path)
        test_data: pd.DataFrame = self.load_csv_file(self.test_data_path)
        # data preprocessing
        self.processor = TabularDataProcessor(self.config)
        self.train_data = self.processor.preprocessing(train_data)
        self.test_data = self.processor.preprocessing(test_data)

    def setup(self):
        # split data based on validation startegy
        splitter = TabularDataSplitter(self.config)
        train_data, valid_data = splitter.split_data(self.train_data)
        # feature engineering
        if self.cv_strategy == 'holdout':
            self.train_data = self.processor.feature_engineering(train_data)
            self.valid_data = self.processor.feature_engineering(valid_data)

            self.train_dataset = TabularDataset(self.train_data)
            self.valid_dataset = TabularDataset(self.valid_data, is_train=False)

        elif self.cv_strategy == 'kfold':
            self.train_data = [self.processor.feature_engineering(df) for df in train_data]
            self.valid_data = [self.processor.feature_engineering(df) for df in valid_data]

            self.train_dataset = [TabularDataset(df) for df in self.train_data]
            self.valid_dataset = [TabularDataset(df, is_train=False) for df in self.valid_data]

        else:
            raise NotImplementedError

        self.test_data = self.processor.feature_engineering(self.test_data)
        self.test_dataset = TabularDataset(self.test_data, is_train=False)

    def load_csv_file(self, path: str) -> pd.DataFrame:
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
            } 
        return pd.read_csv(path, dtype=dtype, parse_dates=['Timestamp'])


class TabularDataProcessor:
    def __init__(self, config: DictConfig):
        self.config = config
        
    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO
        """
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        # userID별 문제 풀이 수
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        # userID별 정답 수
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        # userID별 정답률
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
        # testId별 정답률
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        df = pd.merge(df, correct_t, on=['testId'], how="left")
        # KnowledgeTag별 정답률
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        return df


class TabularDataSplitter:
    def __init__(self, config: DictConfig):
        self.cv_strategy: str = config.cv_strategy
        
    def split_data(self, df: pd.DataFrame, k=5):
        splitter = GroupKFold(n_splits=k)
        train_dataset, valid_dataset = [], []
        for train_index, valid_index in splitter.split(df, groups=df['userID']):
            train_dataset.append(df.loc[train_index])
            valid_dataset.append(df.loc[valid_index])

        if self.cv_strategy == 'holdout':
            return train_dataset[0], valid_dataset[0]

        elif self.cv_strategy == 'kfold':
            return train_dataset, valid_dataset

        else:
            raise NotImplementedError


class TabularDataset:
    def __init__(self, df: pd.DataFrame, is_train=True):
        if is_train == False:
            df = df[df['userID'] != df['userID'].shift(-1)]

        features = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
        self.X = df[features]
        self.y = df['answerCode']
