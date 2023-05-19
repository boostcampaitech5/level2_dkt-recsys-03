import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from omegaconf import OmegaConf
from omegaconf import DictConfig
from sklearn.pipeline import FeatureUnion
from feature.modules import FeatureGenerator
from feature import generators as G


def check_and_get_generators(df: pd.DataFrame, config: DictConfig) -> List[Tuple[str, FeatureGenerator]]:
    features = OmegaConf.to_container(config.features)["features"]
    generators = []
    for feature in features:
        if feature not in df.columns:
            generators.append((feature, get_feature_generator(feature)))
    return generators


def get_feature_engineering_pipeline(generators: List[Tuple[str, FeatureGenerator]]) -> FeatureUnion:
    union = FeatureUnion(generators, verbose=True)
    return union


def get_feature_dtype() -> Dict:
    dtype = {
        # original
        "userID": "category",
        "testId": "category",
        "assessmentItemID": "category",
        "answerCode": "int8",
        "KnowledgeTag": "category",
        # derived
        "UserAcc": "float16",
        "UserTag1Acc": "float16",
        "UserTag2Acc": "float16",
        "UserLastTag1Correct": "int8",
        "UserLastTag2Correct": "int8",
        "ItemNumScaled": "float16",
        "Tag1Acc": "float16",
        "Tag2Acc": "float16",
        "UserTestRetakeCnt": "int8",
        "Tag2": "category",
        "UserRecency1": "int8",
        "UserRecency2": "int8",
        "UserRecency3": "int8",
        "UserRecency4": "int8",
        "UserRecency5": "int8",
    }
    return dtype


def get_feature_dtype_for_lgbm() -> Dict:
    dtype = {
        # original
        "userID": "category",
        "testId": "category",
        "assessmentItemID": "category",
        "answerCode": "int",
        "KnowledgeTag": "category",
        # derived
        "UserAcc": "float",
        "UserTag1Acc": "float",
        "UserTag2Acc": "float",
        "UserLastTag1Correct": "int",
        "UserLastTag2Correct": "int",
        "ItemNumScaled": "float",
        "Tag1Acc": "float",
        "Tag2Acc": "float",
        "UserTestRetakeCnt": "int",
        "Tag2": "category",
        "UserRecency1": "int",
        "UserRecency2": "int",
        "UserRecency3": "int",
        "UserRecency4": "int",
        "UserRecency5": "int",
    }
    return dtype


def validate_generator(feature_generator: FeatureGenerator, df: pd.DataFrame) -> bool:
    result = feature_generator.fit_transform(df)
    return result.shape[0] == len(df) and isinstance(result, np.ndarray)


def get_feature_generator(name: str):
    if name == "UserAcc":
        return G.UserAcc()
    elif name == "UserItemAcc":
        return G.UserItemAcc()
    elif name == "UserTag1Acc":
        return G.UserTag1Acc()
    elif name == "UserTag2Acc":
        return G.UserTag2Acc()
    elif name == "UserLastTag1Correct":
        return G.UserLastTag1Correct()
    elif name == "UserLastTag2Correct":
        return G.UserLastTag2Correct()
    elif name == "UserTestRollingTime":
        return G.UserTestRollingTime()
    elif name == "ItemNumScaled":
        return G.ItemNumScaled()
    elif name == "Tag1Acc":
        return G.Tag1Acc()
    elif name == "Tag2Acc":
        return G.Tag2Acc()
    elif name == "UserTestRetakeCnt":
        return G.UserTestRetakeCnt()
    elif name == "Tag2":
        return G.Tag2()
    elif name == "UserRecency1":
        return G.UserRecency1()
    elif name == "UserRecency2":
        return G.UserRecency2()
    elif name == "UserRecency3":
        return G.UserRecency3()
    elif name == "UserRecency4":
        return G.UserRecency4()
    elif name == "UserRecency5":
        return G.UserRecency5()
    else:
        raise NotImplementedError
