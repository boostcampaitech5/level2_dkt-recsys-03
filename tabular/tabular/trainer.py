import os
import pickle
import wandb
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Tuple
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tabular.data import TabularDataModule
from tabular.data import TabularDataset
from tabular.metric import get_metric


class Trainer:
    def __init__(self, config: DictConfig, datamodule: TabularDataModule):
        self.config = config

        self.datamodule: TabularDataModule = datamodule

        self.train_dataset: TabularDataset = datamodule.train_dataset
        self.valid_dataset: TabularDataset = datamodule.valid_dataset
        self.test_dataset: TabularDataset = datamodule.test_dataset

    def get_model_txt_path(self, fold="") -> str:
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.config.timestamp}_{fold}.txt"
        path = os.path.join(directory, filename)
        return path

    def save_model_pkl(self, model, fold="") -> None:
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{fold}.pkl"
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        with open(save_path, "wb") as fw:
            pickle.dump(model, fw)
        wandb.save(save_path)

    def load_model_pkl(self, fold="") -> None:
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{fold}.pkl"
        load_path = os.path.join(directory, filename)

        with open(load_path, "rb") as f:
            model = pickle.load(f)
        return model

    def get_sample_submission_csv(self) -> pd.DataFrame:
        directory = self.config.paths.data_dir
        filename = "sample_submission.csv"
        path = os.path.join(directory, filename)
        return pd.read_csv(path)

    def make_result_df(
        self, user_ids: List, prob: np.ndarray, true: pd.Series
    ) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "userID": user_ids,
                "prob": prob,
                "pred": np.where(prob >= 0.5, 1, 0),
                "true": true,
            }
        )
        return result

    def save_result_csv(
        self, result: pd.DataFrame, fold="", subset: str = "valid"
    ) -> None:
        directory = os.path.join(self.config.paths.output_dir, self.config.timestamp)
        filename = f"{self.config.timestamp}_{subset}_{fold}.csv"
        save_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)
        result.to_csv(save_path, index=False)
        wandb.save(save_path)

    def train(self) -> None:
        train = self.datamodule.train_dataset
        valid = self.datamodule.valid_dataset

        if self.config.model.name == "LGBM":
            lgb_train = lgb.Dataset(train.X, train.y)
            lgb_valid = lgb.Dataset(valid.X, valid.y)

            model = lgb.train(
                params=OmegaConf.to_container(self.config.model.params),
                train_set=lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=["train", "valid"],
                callbacks=[
                    wandb.lightgbm.wandb_callback(),
                    lgb.early_stopping(stopping_rounds=5),
                ],
            )

            save_path = self.get_model_txt_path()
            model.save_model(save_path, num_iteration=model.best_iteration)
            wandb.save(save_path)

            train_prob: np.ndarray = model.predict(
                train.X, num_iteration=model.best_iteration
            )
            valid_prob: np.ndarray = model.predict(
                valid.X, num_iteration=model.best_iteration
            )

            wandb.lightgbm.log_summary(model, save_model_checkpoint=True)

        train_auc, train_acc = get_metric(train.y, train_prob)
        valid_auc, valid_acc = get_metric(valid.y, valid_prob)
        print(
            f"train auc:{train_auc} valid auc:{valid_auc} train acc:{train_acc} valid acc:{valid_acc}"
        )

        result = self.make_result_df(valid.user_id, valid_prob, valid.y)
        self.save_result_csv(result, subset="valid")

    def inference(self, is_submit: bool = False) -> None:
        test = self.datamodule.test_dataset

        if self.config.model.name == "LGBM":
            load_path = self.get_model_txt_path()
            model = lgb.Booster(model_file=load_path)

            test_prob = model.predict(test.X)

        if is_submit == True:
            submission = self.get_sample_submission_csv()
            submission["prediction"] = test_prob
            self.save_result_csv(submission, subset="submission")

        else:
            test_auc, test_acc = get_metric(test.y, test_prob)

            print(f"test auc:{test_auc} test acc:{test_acc}")
            wandb.run.summary["test_auc"] = test_auc
            wandb.log(
                {
                    "confusion matrix": wandb.plot.confusion_matrix(
                        y_true=test.y.reset_index(drop=True),
                        preds=np.where(test_prob >= 0.5, 1, 0),
                        class_names=["0", "1"],
                    )
                }
            )

            result = self.make_result_df(test.user_id, test_prob, test.y)
            self.save_result_csv(result, subset="test")


class CrossValidationTrainer(Trainer):
    def __init__(self, config: DictConfig, datamodule: TabularDataModule):
        super().__init__(config, datamodule)
        self.config = config
        self.datamodule: TabularDataModule = datamodule
        self.train_dataset: List[TabularDataset] = datamodule.train_dataset
        self.valid_datset: List[TabularDataset] = datamodule.valid_dataset
        self.test_dataset: TabularDataset = datamodule.test_dataset

    def cv(self) -> None:
        cv_score = 0
        for i, (train, valid) in enumerate(zip(self.train_dataset, self.valid_dataset)):
            if self.config.model.name == "LGBM":
                lgb_train = lgb.Dataset(train.X, train.y)
                lgb_valid = lgb.Dataset(valid.X, valid.y)

                model = lgb.train(
                    params=OmegaConf.to_container(self.config.model.params),
                    train_set=lgb_train,
                    num_boost_round=20,
                    valid_sets=[lgb_train, lgb_valid],
                    valid_names=["train", "valid"],
                    callbacks=[
                        wandb.lightgbm.wandb_callback(),
                        lgb.early_stopping(stopping_rounds=5),
                    ],
                )

                save_path = self.get_model_txt_path(fold=str(i))
                model.save_model(save_path, num_iteration=model.best_iteration)
                wandb.save(save_path)

                train_prob: np.ndarray = model.predict(
                    train.X, num_iteration=model.best_iteration
                )
                valid_prob: np.ndarray = model.predict(
                    valid.X, num_iteration=model.best_iteration
                )

                train_auc, train_acc = get_metric(train.y, train_prob)
                valid_auc, valid_acc = get_metric(valid.y, valid_prob)
                print(
                    f"fold: {i} train auc:{train_auc} valid auc:{valid_auc} train acc:{train_acc} valid acc:{valid_acc}"
                )

                cv_score += valid_auc / 5

            if i == 4:
                wandb.lightgbm.log_summary(model, save_model_checkpoint=True)

            result = self.make_result_df(valid.user_id, valid_prob, valid.y)
            self.save_result_csv(result, fold=str(i), subset="valid")

        print(f"cv_score:{cv_score}")
        wandb.log({"cv_score": cv_score})

    def oof(self, is_submit: bool = False) -> None:
        test = self.test_dataset
        probs = []
        for i in range(5):
            if self.config.model.name == "LGBM":
                load_path = self.get_model_txt_path(fold=str(i))
                model = lgb.Booster(model_file=load_path)

                test_prob = model.predict(test.X)
                probs.append(test_prob)
        test_prob, test_pred = self.soft_voting(np.array(probs))

        if is_submit == True:
            submission = self.get_sample_submission_csv()
            submission["prediction"] = test_prob
            self.save_result_csv(submission, subset="submission")

        else:
            test_auc, test_acc = get_metric(test.y, test_prob)

            print(f"test auc:{test_auc} test acc:{test_acc}")
            wandb.run.summary["test_auc"] = test_auc
            wandb.log(
                {
                    "confusion matrix": wandb.plot.confusion_matrix(
                        y_true=test.y.reset_index(drop=True),
                        preds=test_pred,
                        class_names=["0", "1"],
                    )
                }
            )

            result = self.make_result_df(test.user_id, test_prob, test.y)
            self.save_result_csv(result, subset="test")

    def soft_voting(self, probs: np.ndarray) -> Tuple:
        test_prob = np.mean(probs, axis=0)
        test_pred = np.where(test_prob >= 0.5, 1, 0)
        return test_prob, test_pred
