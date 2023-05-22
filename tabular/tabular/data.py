import os
import pandas as pd
from tqdm import tqdm
from typing import List, Union, Optional
from omegaconf import OmegaConf
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
from feature.utils import check_and_get_generators, get_feature_engineering_pipeline, get_feature_dtype, get_feature_dtype_for_lgbm


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.latest_version = True
        self.config = config
        self.data_dir = config.paths.data_dir
        self.data_version = config.data_version
        self.cv_strategy: str = config.cv_strategy

        self.train_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.valid_data: Union[pd.DataFrame, List[pd.DataFrame], None] = None
        self.test_data: Optional[pd.DataFrame] = None

        self.train_dataset: Optional[TabularDataset] = None
        self.valid_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None
        self.ens_dataset: Optional[TabularDataset] = None

    def prepare_data(self):
        """
        데이터 준비
        - 데이터 파일을 불러옵니다.
        - 기본적인 전처리를 수행합니다.
        - _data 인스턴스 변수에 pd.DataFrame 객체 또는 pd.DataFrame 객체 리스트를 할당합니다.
        """
        # load data file
        train_data: pd.DataFrame = self.load_data_file(self.data_dir + "train_data.csv")

        if self.config.is_submit == True:
            test_data: pd.DataFrame = self.load_data_file(self.data_dir + "test_data.csv", is_test=True)
        else:
            test_data: pd.DataFrame = self.load_data_file(self.data_dir + "valid_data.csv", is_test=True)

        # data preprocessing
        self.train_data = self.preprocessing(train_data)
        self.test_data = self.preprocessing(test_data)

    def setup(self):
        """
        데이터 셋업
        - 데이터 분할을 수행합니다.
        - 분할된 각각의 데이터에 파생변수를 생성합니다.
        - _dataset 인스턴스 변수에 Tabular Dataset 객체 또는 Tabular Dataset 객체 리스트를 할당합니다.
        - 데이터 버전을 업데이트 합니다.
        """
        # split data based on validation startegy
        splitter = TabularDataSplitter(self.config)
        train_data, valid_data = splitter.split_data(self.train_data)
        # feature engineering
        if self.cv_strategy == "holdout":
            self.train_data = self.feature_engineering(train_data)
            self.valid_data = self.feature_engineering(valid_data)

            self.train_dataset = TabularDataset(self.config, self.train_data)
            self.valid_dataset = TabularDataset(self.config, self.valid_data)

        elif self.cv_strategy == "kfold":
            self.train_data = [self.feature_engineering(df) for df in train_data]
            self.valid_data = [self.feature_engineering(df) for df in valid_data]

            self.train_dataset = [TabularDataset(self.config, df) for df in self.train_data]
            self.valid_dataset = [TabularDataset(self.config, df) for df in self.valid_data]

        else:
            raise NotImplementedError

        self.test_data = self.feature_engineering(self.test_data)
        self.test_dataset = TabularDataset(self.config, self.test_data, is_test=True)
        self.ens_dataset = TabularDataset(self.config, self.test_data, is_ens=True)
        # update data version
        if self.latest_version == False:
            print("Update data version...")
            self.update_version()

    def load_data_file(self, path: str, fold=0, is_test=False) -> pd.DataFrame:
        """
        데이터 파일 불러오기
        - 데이터 파일을 불러옵니다.
        - uitls.get_feature_dtype 메서드를 사용해서 dtype을 설정하여 메모리를 관리합니다.
        """
        if self.data_version:
            tokens = path.split("data/")
            path = "".join(tokens[0] + "data/" + str(self.data_version) + f"/fold{fold}/" + tokens[-1])

            if is_test == True:
                path = "".join(tokens[0] + "data/" + str(self.data_version) + "/" + tokens[-1])

        if os.path.splitext(path)[1] == ".csv":
            return pd.read_csv(path, dtype=get_feature_dtype(), parse_dates=["Timestamp"])

        else:
            raise NotImplementedError

    def write_metadata(self, df: pd.DataFrame, dir: str) -> None:
        """
        메타데이터 텍스트 파일 쓰기
        - 업데이트된 데이터 버전에 대응하는 metadata.txt 파일을 저장합니다.
        """
        path = dir + "metadata.txt"
        with open(path, "w") as f:
            f.write("Metadata Information:\n")
            f.write("---------------------\n")
            f.write(f"Data Version: {self.config.timestamp}\n")
            f.write(f"IsSubmit: {self.config.is_submit}\n")
            f.write(f'Column Names: {", ".join(df.columns)}\n')

    def update_version(self) -> None:
        """
        데이터 버전 업데이트
        - latest version이 False인 경우 진행합니다.
        - 생성한 파생변수가 추가된 데이터 프레임을 지정한 경로에 csv 파일로 저장합니다.
        """
        data_dir = self.config.paths.data_dir
        version = self.config.timestamp

        if self.cv_strategy == "holdout":
            # sort data
            self.train_data.sort_values(by=["userID", "Timestamp"], inplace=True)
            self.valid_data.sort_values(by=["userID", "Timestamp"], inplace=True)
            # save updated data files
            dir = f"{data_dir}{version}/fold0/"
            os.makedirs(dir, exist_ok=True)
            self.train_data.to_csv(f"{dir}train_data.csv", index=False)
            self.valid_data.to_csv(f"{dir}valid_data.csv", index=False)

        elif self.cv_strategy == "kfold":
            for i in range(self.config.k):
                # sort data
                self.train_data[i].sort_values(by=["userID", "Timestamp"], inplace=True)
                self.valid_data[i].sort_values(by=["userID", "Timestamp"], inplace=True)
                # save updated data files
                dir = f"{data_dir}{version}/fold{i}/"
                os.makedirs(dir, exist_ok=True)
                self.train_data[i].to_csv(f"{dir}train_data.csv", index=False)
                self.valid_data[i].to_csv(f"{dir}valid_data.csv", index=False)

        else:
            raise NotImplementedError

        dir = f"{data_dir}{version}/"
        os.makedirs(dir, exist_ok=True)
        self.test_data.sort_values(by=["userID", "Timestamp"], inplace=True)
        self.test_data.to_csv(f"{dir}test_data.csv", index=False)

        self.write_metadata(self.test_data, dir)

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전처리
        - 이상치 제거
        """
        df = df[df["userID"] != "481"].reset_index(drop=True)
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        파생변수 생성
        - 입력 데이터 컬럼과 config의 feature 목록을 비교하여 생성이 필요한 파생변수 목록을 얻습니다.
        - 파생변수 목록에 해당하는 Feature Generator 객체를 파이프라인에 추가합니다.
        - 파생변수들을 생성하고 입력 데이터에 결합합니다.
        - 작업을 하나라도 수행하는 경우 데이터 버전을 업데이트합니다.
        """
        generators = check_and_get_generators(df, self.config)
        if generators:
            self.latest_version = False
            print("Generate features...")
            pipline = get_feature_engineering_pipeline(generators)
            features_df = pd.DataFrame(
                pipline.fit_transform(df),
                columns=list(pipline.named_transformers.keys()),
            )
            df.reset_index(drop=True, inplace=True)
            result = pd.concat([df, features_df], axis=1)
            return result.astype(get_feature_dtype_for_lgbm())

        else:
            print("Data is lastest version...")
            return df

    def shortcut(self):
        """
        데이터 처리 생략
        - config.skip_data_processing가 True인 경우에 진행합니다.
        - config.paths.data_dir에 있는 데이터 파일들을 불러옵니다.
        - 데이터 처리 작업을 생략하고 바로 _data, _dataset 인스턴스 변수들에 값을 할당합니다.
        """
        if self.cv_strategy == "holdout":
            self.train_data = self.load_data_file(self.data_dir + "train_data.csv")
            self.valid_data = self.load_data_file(self.data_dir + "valid_data.csv")

            self.train_dataset = TabularDataset(self.config, self.train_data)
            self.valid_dataset = TabularDataset(self.config, self.valid_data)

        elif self.cv_strategy == "kfold":
            self.train_data = [self.load_data_file(self.data_dir + "train_data.csv", i) for i in range(self.config.k)]
            self.valid_data = [self.load_data_file(self.data_dir + "valid_data.csv", i) for i in range(self.config.k)]

            self.train_dataset = [TabularDataset(self.config, df) for df in self.train_data]
            self.valid_dataset = [TabularDataset(self.config, df) for df in self.valid_data]

        self.test_data = self.load_data_file(self.data_dir + "test_data.csv", is_test=True)
        self.test_dataset = TabularDataset(self.config, self.test_data, is_test=True)
        self.ens_dataset = TabularDataset(self.config, self.test_data, is_ens=True)


class TabularDataSplitter:
    def __init__(self, config: DictConfig):
        self.config = config
        self.cv_strategy: str = config.cv_strategy

    def split_data(self, df: pd.DataFrame):
        """
        데이터 분할
        - train_data를 입력받아 config.cv_startegy에 기반하여 데이터 분할을 수행합니다.
        - GroupKFold를 사용하여 유저 단위로 데이터를 분할 합니다.
        """
        splitter = GroupKFold(n_splits=self.config.k)
        train_dataset, valid_dataset = [], []
        for train_index, valid_index in splitter.split(df, groups=df["userID"]):
            train_dataset.append(df.loc[train_index])
            valid_dataset.append(df.loc[valid_index])

        if self.cv_strategy == "holdout":
            return train_dataset[0], valid_dataset[0]

        elif self.cv_strategy == "kfold":
            return train_dataset, valid_dataset

        else:
            raise NotImplementedError


class TabularDataset:
    def __init__(self, config: DictConfig, df: pd.DataFrame, is_test=False, is_ens=False):
        if is_test == True:
            df = df.groupby("userID").nth(-1)
        elif is_ens:
            df = df.groupby("userID").nth(-2)

        self.user_id = df["userID"]
        self.X = df[OmegaConf.to_container(config.features)["features"]]
        self.y = df["answerCode"]
