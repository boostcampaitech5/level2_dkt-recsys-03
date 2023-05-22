import os
import numpy as np
import pandas as pd

from .utils import rmse


class Stacking:
    def __init__(self, filenames: list, filepath: str):
        self.filenames = filenames
        self.filepath = filepath

    def get_output_frame(self) -> pd.DataFrame:
        output_frame = pd.read_csv(os.path.join(self.filepath, self.filenames[0]))
        output_frame["prediction"] = output_frame["prediction"].apply(lambda x: 0)
        return output_frame

    def load_train_data(self):
        """
        각 모델의 valid 데이터에 대한 예측 데이터를 불러와 저장합니다.
        """
        valid_path = [self.filepath + filename + "_valid.csv" for filename in self.filenames]
        self.valid_labels = pd.read_csv(valid_path[0])["answer"]
        self.valid_pred_list = []

        for path in valid_path:
            self.valid_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def train(self):
        # prepare_train_data(self):
        pass

    def get_weights(self):
        pass

    def get_bias(self):
        pass

    def load_test_data(self):
        pass

    def infer(self):
        pass
