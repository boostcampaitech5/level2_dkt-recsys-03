import numpy as np
import pandas as pd
import math


class Ensemble:
    def __init__(self, filenames: list, filepath: str):
        self.filenames = filenames
        self.filepath = filepath

        self.load_submit_data()

    def load_submit_data(self):
        submit_path = [self.filepath + filename + "_submit.csv" for filename in self.filenames]

        self.submit_frame = pd.read_csv(submit_path[0])
        self.submit_frame["prediction"] = self.submit_frame["prediction"].apply(lambda x: 0)

        self.submit_pred_list = []
        for path in submit_path:
            self.submit_pred_list.append(pd.read_csv(path)["prediction"].to_list())

    def simple_weighted(self, weight: list):
        """
        직접 weight를 지정하여 앙상블을 수행
        """
        self.weight = weight
        if not len(self.submit_pred_list) == len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if not math.isclose(np.sum(weight), 1):
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.submit_pred_list[0]], [self.submit_pred_list[1]], axis=0)
        for i in range(2, len(self.submit_pred_list)):
            pred_arr = np.append(pred_arr, [self.submit_pred_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()

    def average_weighted(self):
        """
        (1/n)의 동일한 weight로 앙상블을 수행
        """
        self.weight = [1 / len(self.submit_pred_list) for _ in range(len(self.submit_pred_list))]
        pred_weight_list = [pred * np.array(w) for pred, w in zip(self.submit_pred_list, self.weight)]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()

    def set_filename(self):
        weights_info = "-".join([str(w)[:4] for w in self.weight])
        file_title = "-".join(self.filenames)
        return f"{weights_info}-{file_title}"
