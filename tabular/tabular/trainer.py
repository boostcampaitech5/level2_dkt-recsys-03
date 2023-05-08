import os
import pickle
import wandb
import numpy as np
import pandas as pd
import lightgbm as lgb
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tabular.data import TabularDataModule
from tabular.data import TabularDataset


class Trainer:
    def __init__(self, config: DictConfig, datamodule: TabularDataModule):
        self.config = config
        self.model: Opional[lgb.LGBMClassifier] = None

        self.datamodule: TabularDataModule = datamodule

        self.train_dataset: TabularDataset = datamodule.train_dataset
        self.valid_dataset: TabularDataset = datamodule.valid_dataset
        self.test_dataset: TabularDataset = datamodule.test_dataset

    def get_model(self):
        name = self.config.model.name
        params = self.config.model.params

        if name == 'LGBM':
            return lgb.LGBMClassifier(**params)
        else:
            raise NotImplementedError

    def save_model_pkl(self, model, fold=""):
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{fold}.pkl"
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        with open(save_path, 'wb') as fw:
            pickle.dump(model, fw)
        
        # wandb saving
        wandb.save(save_path)
    
    def load_model_pkl(self, fold=""):
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{fold}.pkl"
        load_path = os.path.join(directory, filename)

        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def save_result_csv(self, result: pd.DataFrame, fold="", type: str = 'valid'):
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{type}_{fold}.csv"
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        result.to_csv(save_path, index=False)
        
        # wandb saving
        wandb.save(save_path)

    def train(self):
        model = self.get_model()
        train = self.datamodule.train_dataset
        valid = self.datamodule.valid_dataset

        model.fit(train.X, train.y)
        self.save_model_pkl(model)

        p_train: np.ndarray = model.predict(train.X)
        p_valid: np.ndarray = model.predict(valid.X)

        train_auc = roc_auc_score(train.y, p_train)
        train_acc = accuracy_score(train.y, p_train)

        valid_auc = roc_auc_score(valid.y, p_valid)
        valid_acc = accuracy_score(valid.y, p_valid)

        print(f"fold:{i} train auc:{train_auc} valid auc:{valid_auc} train acc:{train_acc} valid acc:{valid_acc}")
        
        # wandb logging
        wandb.log({"fold":i, "train auc" : train_auc, "valid auc" : valid_auc, "train acc" : train_acc, "valid acc" : valid_acc})

        pb_valid: np.ndarray = model.predict_proba(valid.X)[:, 1]

        user_id = self.datamodule.valid_data['userID'].unique()
        result = pd.DataFrame({'userID': user_id, 'prob': pb_valid, 'pred': p_valid, 'true': valid.y})
        self.save_result_csv(result, 'valid')

    def inference(self, is_submit: bool = False):
        test = self.datamodule.test_dataset

        model = self.load_model_pkl()
        p_test = model.predict(test.X)
        pb_test = model.predict_proba(test.X)[:, 1]
        
        if is_submit == True:
            data_dir = self.config.paths.data_dir
            submission = pd.read_csv(data_dir+'sample_submission.csv')
            submission['prediction'] = p_test
            self.save_result_csv(submission, 'submission')
            
        else:
            test_auc = roc_auc_score(test.y, p_test)
            test_acc = accuracy_score(test.y, p_test)
            print(f"test auc:{test_auc} test acc:{test_acc}")
            
            # wandb logging
            wandb.log({"test auc" : test_auc, "test acc" : test_acc})

            user_id = self.datamoudle.test_data['userID'].unique()
            result = pd.DataFrame({'userID': user_id, 'prob': pb_test, 'pred': p_test, 'true': test.y})
            self.save_result_csv(result, 'test')


class CrossValidationTrainer(Trainer):
    def __init__(self, config: DictConfig, datamodule: TabularDataModule):
        super().__init__(config, datamodule)
        self.config = config
        self.datamodule: TabularDataModule = datamodule
        self.train_dataset: List[TabularDataset] = datamodule.train_dataset
        self.valid_datset: List[TabularDataset] = datamodule.valid_dataset
        self.test_dataset: TabularDataset = datamodule.test_dataset
        
    def cv(self):
        for i, (train, valid) in enumerate(zip(self.train_dataset, self.valid_dataset)):
            model = self.get_model()
            model.fit(train.X, train.y)
            self.save_model_pkl(model, fold=str(i))
            p_train: np.ndarray = model.predict(train.X)
            p_valid: np.ndarray = model.predict(valid.X)
        
            train_auc = roc_auc_score(train.y, p_train)
            train_acc = accuracy_score(train.y, p_train)

            valid_auc = roc_auc_score(valid.y, p_valid)
            valid_acc = accuracy_score(valid.y, p_valid)

            print(f"fold:{i} train auc:{train_auc} valid auc:{valid_auc} train acc:{train_acc} valid acc:{valid_acc}")

            # wandb logging
            wandb.log({"fold":i, "train auc" : train_auc, "valid auc" : valid_auc, "train acc" : train_acc, "valid acc" : valid_acc})

            pb_valid: np.ndarray = model.predict_proba(valid.X)[:, 1]

            user_id = self.datamodule.valid_data[i]['userID'].unique()
            result = pd.DataFrame({'userID': user_id, 'prob': pb_valid, 'pred': p_valid, 'true': valid.y})
            self.save_result_csv(result, fold=str(i), type='valid')
    
    def oof(self, is_submit: bool = False):
        test = self.test_dataset
        pred = []
        for i in range(5):
            model = self.load_model_pkl(fold=str(i))
            pb_test = model.predict_proba(test.X)[:, 1]
            pred.append(pb_test)
        pb_test, p_test = self.soft_voting(np.array(pred))
        
        if is_submit == True:
            data_dir = self.config.paths.data_dir
            submission = pd.read_csv(data_dir+'sample_submission.csv')
            submission['prediction'] = p_test
            self.save_result_csv(submission, type='submission')
            
        else:
            test_auc = roc_auc_score(test.y, p_test)
            test_acc = accuracy_score(test.y, p_test)
            print(f"test auc:{test_auc} test acc:{test_acc}")
            wandb.log({"test auc" : test_auc, "test acc" : test_acc})

            user_id = self.datamodule.test_data['userID'].unique()
            result = pd.DataFrame({'userID': user_id, 'prob': pb_test, 'pred': p_test, 'true': test.y})
            self.save_result_csv(result, type='test')

    def soft_voting(self, pred: np.ndarray):
        pb_test = np.mean(pred, axis=0)
        p_test = np.where(pb_test >= 0.5, 1, 0)
        return pb_test, p_test
      