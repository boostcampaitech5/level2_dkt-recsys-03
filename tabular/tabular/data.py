import random
import pandas as pd
from typing import Dict, List, Union, Optional
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.train_data_path: str = config.train_data_path
        self.test_data_path: str = config.test_data_path
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
            train_data = self.processor.feature_engineering(self.train_data)
            valid_data = self.processor.feature_engineering(valid_data)

            self.train_dataset = TabularDataset(self.train_data)
            self.valid_dataset = TabularDataset(valid_data, is_train=False)

        elif self.cv_strategy == 'kfold':
            train_data = [self.processor.feature_engineering(df) for df in train_data]
            valid_data = [self.processor.feature_engineering(df) for df in valid_data]

            self.train_dataset = [TabularDataset(df) for df in train_data]
            self.valid_dataset = [TabularDataset(df) for df in valid_data]

        else:
            raise NotImplementedError

        test_data = self.processor.feature_engineering(self.test_data)
        self.test_dataset = TabularDataset(test_data)

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
        """
        TODO
        """ 
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

        self.X = df.drop(['answerCode'], axis=1)
        self.y = df['answerCode']