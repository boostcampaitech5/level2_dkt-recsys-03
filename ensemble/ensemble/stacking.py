import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .utils import rmse


class Stacking:
    def __init__(self, filenames: list, filepath: str, seed: int, test_size: float):
        self.filenames = filenames
        self.filepath = filepath
        self.seed = seed
        self.test_size = test_size

        self.model = LinearRegression()  # stacking model

        self.load_valid_data()
        self.load_submit_data()

    def load_valid_data(self):
        valid_path = [self.filepath + filename + "_valid.csv" for filename in self.filenames]

        self.valid_labels = pd.read_csv(valid_path[0])["answer"].to_list()
        self.valid_pred_list = []
        for path in valid_path:
            self.valid_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def train(self):
        X = np.transpose(self.valid_pred_list)
        y = self.valid_labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

        self.model.fit(X_train, y_train)  # training model

        test_pred = self.model.predict(X_test)  # testing model
        loss = rmse(test_pred, y_test)

        print(f"Weight: {self.get_weights()}")
        print(f"Bias: {self.get_bias()}")
        print(f"Train RMSE: {loss}")

    def get_weights(self):
        return self.model.coef_

    def get_bias(self):
        return self.model.intercept_

    def load_submit_data(self):
        submit_path = [self.filepath + filename + "_submit.csv" for filename in self.filenames]

        self.submit_frame = pd.read_csv(submit_path[0])
        self.submit_frame["prediction"] = self.submit_frame["prediction"].apply(lambda x: 0)

        self.submit_pred_list = []
        for path in submit_path:
            self.submit_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def infer(self):
        X = np.transpose(self.submit_pred_list)
        pred = self.model.predict(X)
        return pred
