import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from data import TabularDataModule
from data import TabularDataset


class Trainer:
    def __init__(self, config: DictConfig, datamodule: TabularDataModule):
        self.config = config
        self.model: Opional[lgb.LGBMClassifier] = None

        self.datamodule: TabularDataModule = datamodule

        self.train_dataset: TabularDataset = datamodule.train_dataset
        self.valid_dataset: TabularDataset = datamodule.valid_dataset
        self.test_dataset: TabularDataset = datamodule.test_dataset

        self.x_train: pd.DataFrame = self.train_dataset.X
        self.y_train: pd.Series = self.train_dataset.y

        self.x_valid: pd.DataFrame = self.valid_dataset.X
        self.y_valid: pd.Series = self.valid_dataset.y

        self.x_test: pd.DataFrame = self.test_dataset.X
        self.y_test: pd.Series = self.test_dataset.y

    def init_model(self):
        name = self.config.model.name
        params = self.config.model.params

        if name == 'LGBM':
            self.model = lgb.LGBMClassifier(**params)
        else:
            raise NotImplementedError

    def save_model_pkl(self, model):
        directory = os.path.join(self.config.output_dir, self.config.timestamp)
        filename = self.config.timestamp+'.pkl'
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        with open(save_path, 'wb') as fw:
            pickle.dump(model, fw)
    
    def load_model_pkl(self):
        directory = os.path.join(self.config.output_dir, self.config.timestamp)
        filename = self.config.timestamp+'.pkl'
        load_path = os.path.join(directory, filename)

        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def save_result_csv(self, result: pd.DataFrame, type: str):
        directory = os.path.join(self.config.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{type}.csv"
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        result.to_csv(save_path, index=False)

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        self.save_model_pkl(self.model)

        p_train: np.ndarray = self.model.predict(self.x_train)
        p_valid: np.ndarray = self.model.predict(self.x_valid)

        train_auc = roc_auc_score(self.y_train, p_train)
        train_acc = accuracy_score(self.y_train, p_train)

        valid_auc = roc_auc_score(self.y_valid, p_valid)
        valid_acc = accuracy_score(self.y_valid, p_valid)

        print(f"train auc:{train_auc} train acc:{train_acc}")
        print(f"valid auc:{valid_auc} valid acc:{valid_acc}")

        pb_valid: np.ndarray = self.model.predict_proba(self.x_valid)[:, 1]

        user_id = self.datamodule.valid_data['userID'].unique()
        result = pd.DataFrame({'userID': user_id, 'prob': pb_valid, 'pred': p_valid, 'true': self.y_valid})
        self.save_result_csv(result, 'valid')

    def inference(self, is_submit: bool = True):
        model = self.load_model_pkl()
        p_test = model.predict(self.x_test)
        pb_test = model.predict_proba(self.x_test)[:, 1]
        
        if is_submit == True:
            data_dir = self.config.data_dir
            submission = pd.read_csv(data_dir+'sample_submission.csv')
            submission['prediction'] = p_test
            self.save_result_csv(submission, 'submission')
            
        else:
            test_auc = roc_auc_score(self.y_test, p_test)
            test_acc = accuracy_score(self.y_test, p_test)
            print(f"test auc:{test_auc} test acc:{test_acc}")

            user_id = self.datamoudle.test_data['userID'].unique()
            result = pd.DataFrame({'userID': user_id, 'prob': pb_test, 'pred': p_test, 'true': self.y_test})
            self.save_result_csv(result, 'test')
