import os
import wandb
import torch
import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig
from sklearn.model_selection import KFold

from .dataloader import DKTDataModule, DKTDataKFoldModule
from .models import LSTM, LSTMATTN, GRUATTN, BERT, LQTR, SAINTPLUS, GPT2
from .utils import get_logger, logging_conf


logger = get_logger(logging_conf)


class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config

    def load_data(self, mode: str = "submit"):
        print(f"----------------- Setup datamodule -----------------")
        logger.info("Preparing data ...")
        return DKTDataModule(self.config, mode)

    def load_model(self):
        logger.info("Building Model ...")
        if self.config.model.model_name == "LSTM":
            wandb.save(f"./configs/model/LSTM.yaml")
            return LSTM(self.config)
        elif self.config.model.model_name == "LSTMATTN":
            wandb.save(f"./configs/model/LSTMATTN.yaml")
            return LSTMATTN(self.config)
        elif self.config.model.model_name == "GRUATTN":
            wandb.save(f"./configs/model/GRUATTN.yaml")
            return GRUATTN(self.config)
        elif self.config.model.model_name == "BERT":
            wandb.save(f"./configs/model/BERT.yaml")
            return BERT(self.config)
        elif self.config.model.model_name == "SASREC":
            wandb.save(f"./configs/model/SASREC.yaml")
            return BERT(self.config)
        elif self.config.model.model_name == "LQTR":
            wandb.save(f"./configs/model/LQTR.yaml")
            return LQTR(self.config)
        elif self.config.model.model_name == "SAINT_PLUS":
            wandb.save(f"./configs/model/SAINT_PLUS.yaml")
            return SAINTPLUS(self.config)
        elif self.config.model.model_name == "GPT2":
            wandb.save(f"./configs/model/GPT2.yaml")
            return GPT2(self.config)
        else:
            raise Exception(f"Wrong model name is used : {self.config.model.model_name}")

    def write_result_csv(self, pred: np.ndarray, mode: str, true=None, fold: int = None, oof: bool = False) -> None:
        result_df = pd.DataFrame(pred).reset_index()
        result_df.columns = ["id", "prediction"]

        if mode == "valid":
            result_df["answer"] = true

        if oof == True:
            if mode == "submit":
                file_name = self.config.wandb.name + "_" + self.config.model.model_name + str(self.config.trainer.k) + "final_submit.csv"
            elif mode == "valid":
                file_name = self.config.wandb.name + "_" + self.config.model.model_name + str(self.config.trainer.k) + "final_valid.csv"
        else:
            if fold == None:
                if mode == "submit":
                    file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_submit.csv"
                elif mode == "valid":
                    file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_valid.csv"

            else:
                if mode == "submit":
                    file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_" + str(fold) + "_submit.csv"
                elif mode == "valid":
                    file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_" + str(fold) + "_valid.csv"

        write_path = os.path.join(self.config.paths.output_path, file_name)
        os.makedirs(name=self.config.paths.output_path, exist_ok=True)
        result_df.to_csv(write_path, index=False)
        print(f"Successfully saved submission as {write_path}")
        wandb.save(write_path)

    def train(self):
        early_stop_callback = EarlyStopping(monitor="val_auc", patience=self.config.trainer.patience, verbose=True, mode="max")
        self.trainer = pl.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[early_stop_callback])

        self.dm = self.load_data()
        self.model = self.load_model()

        logger.info("Start Training ...")
        self.trainer.fit(self.model, datamodule=self.dm)

    def predict(self):
        self.sub_dm = self.load_data(mode="submit")
        self.val_dm = self.load_data(mode="valid")
        logger.info("Making Prediction ...")
        sub_predictions = self.trainer.predict(self.model, datamodule=self.sub_dm)
        val_predictions = self.trainer.predict(self.model, datamodule=self.val_dm)
        val_true = self.val_dm.test_answercode

        logger.info("Saving Submission ...")
        self.write_result_csv(np.concatenate(sub_predictions), mode="submit")
        self.write_result_csv(np.concatenate(val_predictions), mode="valid", true=val_true)


class KfoldTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config
        self.cv_score = 0
        self.sub_result_csv_list = []
        self.val_result_csv_list = []

    def cv(self):
        kf = KFold(
            n_splits=self.config.trainer.k,
            random_state=self.config.seed,
            shuffle=True,
        )

        # load original data and groupby user and preprocessing
        self.sub_dm = self.load_data(mode="submit")
        self.sub_dm.prepare_data()
        self.sub_dm.setup()

        self.val_dm = self.load_data(mode="valid")
        self.val_dm.prepare_data()
        self.val_dm.setup()

        # load train dataset
        tr_dataset = self.sub_dm.train_data
        val_dastaset = self.sub_dm.valid_data
        tr_dataset = np.concatenate((tr_dataset, val_dastaset), axis=0)  # concat for k-fold cv
        test_dataset = self.sub_dm.test_data

        # K-fold Cross Validation
        for fold, (tra_idx, val_idx) in enumerate(kf.split(tr_dataset)):
            print(f"------------- Fold {fold}  :  train {len(tra_idx)}, val {len(val_idx)} -------------")
            self.config.trainer.total_steps = math.ceil(len(tra_idx) / self.config.data.batch_size) * self.config.trainer.epoch
            self.config.trainer.warmup_steps = self.config.trainer.total_steps // 10

            # create model for cv
            self.fold_model = self.load_model()
            # set data for training and validation in fold
            early_stop_callback = EarlyStopping(monitor="val_auc", patience=self.config.trainer.patience, verbose=True, mode="max")
            self.fold_trainer = pl.Trainer(max_epochs=self.config.trainer.epoch, callbacks=[early_stop_callback])

            self.fold_dm = DKTDataKFoldModule(self.config, mode="submit")
            self.fold_dm.train_data = tr_dataset[tra_idx]
            self.fold_dm.valid_data = tr_dataset[val_idx]
            self.fold_dm.test_data = test_dataset

            # train n validation
            self.fold_trainer.fit(self.fold_model, datamodule=self.fold_dm)

            print(
                "check tr_result, val_result: ",
                len(self.fold_model.tr_result),
                len(self.fold_model.val_result),
            )
            tr_auc = torch.stack([x["tr_avg_auc"] for x in self.fold_model.tr_result]).mean()
            tr_acc = torch.stack([x["tr_avg_acc"] for x in self.fold_model.tr_result]).mean()
            val_auc = torch.stack([x["val_avg_auc"] for x in self.fold_model.val_result]).mean()
            val_acc = torch.stack([x["val_avg_acc"] for x in self.fold_model.val_result]).mean()

            print(f">>> >>> tr_auc: {tr_auc}, tr_acc: {tr_acc}, val_auc: {val_auc}, val_acc: {val_acc}")
            self.cv_score += val_auc / self.config.trainer.k
            self.cv_predict(fold)

        # cv_score result
        print(f"-----------------cv_auc_score: {self.cv_score}-----------------")
        wandb.log({"cv_score": self.cv_score})

    def cv_predict(self, fold: int):
        logger.info("Making Prediction ...")
        sub_predictions = self.fold_trainer.predict(self.fold_model, datamodule=self.fold_dm)
        val_predictions = self.fold_trainer.predict(self.fold_model, datamodule=self.val_dm)
        val_true = self.val_dm.test_answercode

        logger.info("Saving Submission ...")
        self.write_result_csv(np.concatenate(sub_predictions), fold=fold, mode="submit")
        self.write_result_csv(np.concatenate(val_predictions), fold=fold, mode="valid", true=val_true)
        self.set_result_csv_list(fold=fold, mode="submit")
        self.set_result_csv_list(fold=fold, mode="valid")

    def oof(self):
        # load all submission csv files
        print(f"----------------- Load files for OOF -----------------")

        sub_df_list = []
        for file in self.sub_result_csv_list:
            df = pd.read_csv(file)
            sub_df_list.append(df["prediction"])

        val_df_list = []
        for file in self.val_result_csv_list:
            df = pd.read_csv(file)
            val_df_list.append(df["prediction"])

        # soft voting
        sub_predictions, _ = self.soft_voting(sub_df_list)
        val_predictions, _ = self.soft_voting(val_df_list)
        val_true = self.val_dm.test_answercode

        self.write_result_csv(sub_predictions, oof=True, mode="submit")
        self.write_result_csv(val_predictions, oof=True, mode="valid", true=val_true)

        # save test_prob file in local, wandb -> bar plot
        # save test_pred file in local, wandb -> confusion matrix

    def soft_voting(self, df_list):
        test_prob = np.mean(np.array(df_list), axis=0)
        test_pred = np.where(test_prob >= 0.5, 1, 0)

        return test_prob, test_pred

    def set_result_csv_list(self, fold, mode):
        if mode == "submit":
            file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_" + str(fold) + "_submit.csv"
            write_path = os.path.join(self.config.paths.output_path, file_name)
            self.sub_result_csv_list.append(write_path)
        elif mode == "valid":
            file_name = self.config.wandb.name + "_" + self.config.model.model_name + "_" + str(fold) + "_valid.csv"
            write_path = os.path.join(self.config.paths.output_path, file_name)
            self.val_result_csv_list.append(write_path)
