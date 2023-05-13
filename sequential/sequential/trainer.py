import os
import wandb
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import KFold

from .dataloader import DKTDataModule, DKTDataKFoldModule
from .models import LSTM, LSTMATTN, BERT
from .utils import get_logger, logging_conf


logger = get_logger(logging_conf)


class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config

    def load_data(self):
        print(f"----------------- Setup datamodule -----------------")
        logger.info("Preparing data ...")
        return DKTDataModule(self.config)

    def load_model(self):
        logger.info("Building Model ...")
        if self.config.model.model_name == "LSTM":
            wandb.save(f"./configs/model/LSTM.yaml")
            return LSTM(self.config)
        elif self.config.model.model_name == "LSTMATTN":
            wandb.save(f"./configs/model/LSTMATTN.yaml")
            return LSTMATTN(self.config)
        elif self.config.model.model_name == "BERT":
            wandb.save(f"./configs/model/BERT.yaml")
            return BERT(self.config)
        else:
            raise Exception(f"Wrong model name is used : {self.config.model.model_name}")

    def train(self):
        self.trainer = pl.Trainer(max_epochs = self.config.trainer.epoch)

        self.dm = self.load_data()
        self.model = self.load_model()

        logger.info("Start Training ...")
        self.trainer.fit(self.model, datamodule=self.dm)
    
    def predict(self):
        logger.info("Making Prediction ...")
        predictions = self.trainer.predict(self.model, datamodule=self.dm)

        logger.info("Saving Submission ...")
        predictions = np.concatenate(predictions)
        submit_df = pd.DataFrame(predictions)
        submit_df = submit_df.reset_index()
        submit_df.columns = ['id', 'prediction']

        file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_submit.csv"

        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submit_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)


class KfoldTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config
        self.cv_score = 0
        self.result_csv_list = []
    
    def cv(self):
        kf = KFold(n_splits = self.config.trainer.k, random_state=self.config.seed, shuffle=True)

        # load original data and groupby user and preprocessing
        self.dm = self.load_data()
        self.dm.prepare_data()
        self.dm.setup()

        # load train dataset
        tr_dataset = self.dm.train_data
        val_dastaset = self.dm.valid_data
        tr_dataset = np.concatenate((tr_dataset, val_dastaset), axis=0) # concat for k-fold cv
        test_dataset = self.dm.test_data

        # K-fold Cross Validation
        for fold, (tra_idx, val_idx) in enumerate(kf.split(tr_dataset)):
            print(f"------------- Fold {fold}  :  train {len(tra_idx)}, val {len(val_idx)} -------------")

            # create model for cv
            self.model = self.load_model()

            # set data for training and validation in fold
            self.trainer = pl.Trainer(max_epochs = self.config.trainer.epoch)

            self.dm = DKTDataKFoldModule(self.config)
            self.dm.train_data = tr_dataset[tra_idx]
            self.dm.valid_data = tr_dataset[val_idx]
            self.dm.test_data = test_dataset
            
            # train n validation
            self.trainer.fit(self.model, datamodule=self.dm)

            print("check tr_result, val_result: ", len(self.model.tr_result), len(self.model.val_result))
            tr_auc = torch.stack([x['tr_avg_auc'] for x in self.model.tr_result]).mean()
            tr_acc = torch.stack([x['tr_avg_acc'] for x in self.model.tr_result]).mean()
            val_auc = torch.stack([x['val_avg_auc'] for x in self.model.val_result]).mean()
            val_acc = torch.stack([x['val_avg_acc'] for x in self.model.val_result]).mean()

            print(f">>> >>> tr_auc: {tr_auc}, tr_acc: {tr_acc}, val_auc: {val_auc}, val_acc: {val_acc}")
            self.cv_score += (val_auc / self.config.trainer.k)
            self.cv_predict()
        
        # cv_score result
        print(f"cv_auc_score: {self.cv_score}")
        wandb.log({'cv_score': self.cv_score})

    def cv_predict(self):
        logger.info("Making Prediction ...")
        predictions = self.trainer.predict(self.model, datamodule=self.dm)

        logger.info("Saving Submission ...")
        predictions = np.concatenate(predictions)
        submit_df = pd.DataFrame(predictions)
        submit_df = submit_df.reset_index()
        submit_df.columns = ['id', 'prediction']

        file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_submit.csv"
        write_path = os.path.join(self.config.paths.output_path, file_name)
        self.result_csv_list.append(write_path)

        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submit_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)
    
    def oof(self):
        # load sample files
        sample_path = os.path.join(self.config.paths.data_path, self.config.paths.sample_file)
        submit_df = pd.read_csv(sample_path)

        # load all submission csv files
        print(f"----------------- Load files for OOF -----------------")
        df_list = []
        for file in self.result_csv_list:
            df = pd.read_csv(file)
            df_list.append(df['prediction'])
        
        # soft voting
        submit_df['prediction'] = pd.DataFrame(np.mean(np.array(df_list), axis=0))

        # file saving
        file_name = self.config.wandb.name + "_" + self.config.model.model_name + str(self.config.trainer.k) + "fold_submit.csv"
        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)

        submit_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)

        # cal soft_voting -> test_prob, test_pred

        # save test_prob file in local, wandb -> bar plot
        # save test_pred file in local, wandb -> confusion matrix