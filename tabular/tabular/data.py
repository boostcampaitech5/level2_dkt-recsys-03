import os
import random
import pandas as pd
from typing import Dict, List, Union, Optional
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
from features.feature_context import feature_manager
from features.feature_manager import FeatureManager


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.train_data_path: str = os.path.join(config.paths.data_dir, 'train_data.csv')
        self.test_data_path: str = os.path.join(config.paths.data_dir, 'valid_data.csv')
        self.cv_strategy: str = config.cv_strategy

        self.train_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.valid_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self.train_dataset: Optional[TabularDataset] = None
        self.valid_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None
        
        self.feature_manager: FeatureManager = feature_manager(csv_path=self.config.paths.data_dir + 'feature.csv')

    def prepare_data(self):
        # load csv file
        train_data: pd.DataFrame = self.load_csv_file(self.train_data_path)
        test_data: pd.DataFrame = self.load_csv_file(self.test_data_path)
        # data preprocessing
        self.processor = TabularDataProcessor(self.config, self.feature_manager)
        self.train_data = self.processor.preprocessing(train_data)
        self.test_data = self.processor.preprocessing(test_data)

    def setup(self):
        # split data based on validation startegy
        splitter = TabularDataSplitter(self.config)
        train_data, valid_data = splitter.split_data(self.train_data)
        # feature engineering
        if self.cv_strategy == 'holdout':
            self.train_data = self.processor.feature_engineering(train_data, type='train')
            self.valid_data = self.processor.feature_engineering(valid_data, type='valid')

            self.train_dataset = TabularDataset(self.config, self.train_data)
            self.valid_dataset = TabularDataset(self.config, self.valid_data, is_train=False)

        elif self.cv_strategy == 'kfold':
            self.train_data = [self.processor.feature_engineering(df, type='train', fold=str(i)) for i, df in enumerate(train_data)]
            self.valid_data = [self.processor.feature_engineering(df, type='valid', fold=str(i)) for i, df in enumerate(valid_data)]

            self.train_dataset = [TabularDataset(self.config, df) for df in self.train_data]
            self.valid_dataset = [TabularDataset(self.config, df, is_train=False) for df in self.valid_data]

        else:
            raise NotImplementedError

        self.test_data = self.processor.feature_engineering(self.test_data, type='test')
        self.test_dataset = TabularDataset(self.config, self.test_data, is_train=False)

    def load_csv_file(self, path: str) -> pd.DataFrame:
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
            } 
        return pd.read_csv(path, dtype=dtype, parse_dates=['Timestamp'])


class TabularDataProcessor:
    def __init__(self, config: DictConfig, fm: FeatureManager):
        self.config = config
        self.feature_manager = fm
        
    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        전처리
        - userID, Timestamp 기준 오름차순 정렬
        """
        df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        return df
    
    def feature_engineering(self, df: pd.DataFrame, type: str = 'train', fold: str = "") -> pd.DataFrame:

        if self.feature_manager.need_feature_creation(type=type, fold=fold):
            print(f"Saving features Dataframe csv.. --type {type} --fold {fold}")
            self.feature_manager.create_features(df, type=type, fold=fold)
        
        print(f"Loading features Dataframe csv.. --type {type} --fold {fold}")
        df = self.feature_manager.prepare_df(self.config.features, self.config.features.features, df, type=type, fold=fold)
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
    def __init__(self, config: DictConfig, df: pd.DataFrame, is_train=True):
        if is_train == False:
            df = df[df['userID'] != df['userID'].shift(-1)]

        self.X = df[config.features.features]
        self.y = df['answerCode']
