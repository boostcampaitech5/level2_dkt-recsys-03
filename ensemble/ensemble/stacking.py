import wandb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, cross_val_score

from .utils import get_metric


class StackingBase:
    def __init__(self, filenames: list, filepath: str, seed: int, test_size: float):
        self.filenames = filenames
        self.filepath = filepath
        self.seed = seed
        self.test_size = test_size

        self.load_valid_data()
        self.load_submit_data()

        self.model = LinearRegression()  # stacking model

    def load_valid_data(self):
        valid_path = [self.filepath + filename + "_valid.csv" for filename in self.filenames]

        self.valid_labels = pd.read_csv(valid_path[0])["answer"].to_list()
        self.valid_pred_list = []
        for path in valid_path:
            self.valid_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def load_submit_data(self):
        submit_path = [self.filepath + filename + "_submit.csv" for filename in self.filenames]

        self.submit_frame = pd.read_csv(submit_path[0])
        self.submit_frame["prediction"] = self.submit_frame["prediction"].apply(lambda x: 0)

        self.submit_pred_list = []
        for path in submit_path:
            self.submit_pred_list.append(pd.read_csv(path)["prediction"].to_list())


class Stacking(StackingBase):
    def __init__(self, filenames: list, filepath: str, seed: int, test_size: float):
        super().__init__(filenames, filepath, seed, test_size)

    def fit(self, verbose=True):
        X = np.transpose(self.valid_pred_list)
        y = np.array(self.valid_labels)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

        self.model.fit(X_train, y_train)  # train model

        if verbose:
            print(f"Weight: {self.get_weights()}")
            print(f"Bias: {self.get_bias()}")

        y_pred = self.model.predict(X_valid)  # validation model
        val_auc, val_acc = get_metric(targets=y_valid, preds=y_pred)

        print("========= Holdout Valid =========")
        if verbose:
            print(f">>> AUC: {val_auc}, ACC: {val_acc}")

        wandb.log({"weights": self.get_weights()})
        wandb.log({"val_AUC": val_auc})
        wandb.log({"val_ACC": val_acc})

    def get_weights(self):
        return self.model.coef_

    def get_bias(self):
        return self.model.intercept_

    def infer(self):
        X = np.transpose(self.submit_pred_list)
        pred = self.model.predict(X)
        return pred

    def set_filename(self):
        weights_info = "-".join([str(w)[:4] for w in self.get_weights()])
        file_title = "-".join(self.filenames)
        return f"stack-{weights_info}-{file_title}.csv"


class OofStacking(StackingBase):
    def __init__(self, filenames: list, filepath: str, seed: int, test_size: float, k: int):
        super().__init__(filenames, filepath, seed, test_size)
        self.k = k

    def fit(self, verbose=True):
        X = np.transpose(self.valid_pred_list)
        y = np.array(self.valid_labels)

        self.cv = cross_validate(self.model, X, y, cv=self.k, return_estimator=True)

        aucs = []
        for f_idx, model in enumerate(self.cv["estimator"]):
            pred = model.predict(X)

            val_auc, val_acc = get_metric(targets=y, preds=pred)
            print(f">>> [{f_idx} fold Score] AUC: {val_auc}, ACC: {val_acc}")
            aucs.append(val_auc)

        cv_score = np.array(aucs).mean()
        wandb.log({"cv_score": cv_score})

        if verbose:
            print("========= Cross Validation Valid =========")
            print(f"CV score: {cv_score}")

    def infer(self):
        X_test = np.transpose(self.submit_pred_list)

        preds = []
        for model in self.cv["estimator"]:
            preds.append(model.predict(X_test))
        preds = np.array(preds)

        return preds.mean(axis=0)

    def set_filename(self):
        filename = "-".join(self.filenames)
        return f"oof-{filename}"
