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
        if self.cv_strategy == 'holdout':
            '''
            TODO HoldoutValidationBuilder
            '''
        elif self.cv_strategy == 'kfold':
            '''
            TODO KFoldValidationBuilder
            '''
        else:
            raise NotImplementedError
    
    def load_csv_file(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)