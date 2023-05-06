import random
import pandas as pd
from typing import Dict, Optional, List
from omegaconf import DictConfig


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.train_data_path: str = config.train_data_path
        self.test_data_path: str = config.test_data_path
        self.cv_strategy: str = config.cv_strategy

        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self.train_dataset: Optional[pd.DataFrame] = None
        self.valid_dataset: Optional[pd.DataFrame] = None
        self.test_dataset: Optional[pd.DataFrame] = None

        self.train_datasets: Optional[List[pd.DataFrame]] = None
        self.valid_datasets: Optional[List[pd.DataFrame]] = None

    def prepare_data(self):
        # load csv file
        self.train_data: pd.DataFrame = self.load_csv_file(self.train_data_path)
        self.test_data: pd.DataFrame = self.load_csv_file(self.test_data_path)
        # data preprocessing
        '''
        TODO TabularDataProcessor
        '''

    def setup(self):
        # split data based on validation startegy
        splitter = TabularDataSplitter(self.config)
        if self.cv_strategy == 'holdout':
            self.train_dataset, self.valid_dataset = splitter.split_data(self.train_data)
        elif self.cv_strategy == 'kfold':
            self.train_datasets, self.valid_datasets = splitter.split_data(self.train_data)
        else:
            NotImplementedError
    
    def load_csv_file(self, path: str) -> pd.DataFrame:
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
            } 
        return pd.read_csv(path, dtype=dtype, parse_dates=['Timestamp'])


class TabularDataSplitter:
    def __init__(self, config: DictConfig):
        self.cv_strategy: str = config.cv_strategy
    
    def split_data(self, df: pd.DataFrame, train_size=0.8, shuffle=True):
        if self.cv_strategy == 'holdout':
            users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
            
            if shuffle == True:
                random.shuffle(users)

            max_cnt = train_size*len(df)
            total_cnt = 0
            user_ids = []
            for user_id, cnt in users:
                total_cnt += cnt
                if max_cnt < total_cnt:
                    break
                user_ids.append(user_id)

            train_data = df[df['userID'].isin(user_ids)]
            test_data = df[df['userID'].isin(user_ids) == False]
            return train_data, test_data
    
        else:
            raise NotImplementedError
