import numpy as np
import pandas as pd
import math


class Ensemble:
    def __init__(self, filenames: list, filepath: str):
        self.filenames = filenames
        self.filepath = filepath
        self.is_weighted = True

        self.load_submit_data()

    def load_submit_data(self):
        submit_path = [self.filepath + filename + "_submit.csv" for filename in self.filenames]

        self.submit_frame = pd.read_csv(submit_path[0])
        self.submit_frame["prediction"] = self.submit_frame["prediction"].apply(lambda x: 0)

        self.submit_pred_list = []
        for path in submit_path:
            self.submit_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def average_weighted(self):
        self.weight = [1 / len(self.submit_pred_list) for _ in range(len(self.submit_pred_list))]
        pred_weight_list = [pred * np.array(w) for pred, w in zip(self.submit_pred_list, self.weight)]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()

    def set_filename(self):
        if self.is_weighted:
            weights_info = "-".join([str(w)[:4] for w in self.weight])
            file_title = "-".join(self.filenames)
            return f"{weights_info}-{file_title}.csv"
        else:
            file_title = "-".join(self.filenames)
            return f"{file_title}.csv"

    def mixed(self):
        self.is_weighted = False
