import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from omegaconf import OmegaConf
from omegaconf import DictConfig
from sklearn.pipeline import FeatureUnion
from feature.modules import FeatureGenerator
from feature import generators as G


def check_and_get_generators(
    df: pd.DataFrame, config: DictConfig
) -> List[Tuple[str, FeatureGenerator]]:
    features = OmegaConf.to_container(config.features)["features"]
    generators = []
    for feature in features:
        if feature not in df.columns:
            generators.append((feature, get_feature_generator(feature)))
    return generators


def get_feature_engineering_pipeline(
    generators: List[Tuple[str, FeatureGenerator]]
) -> FeatureUnion:
    union = FeatureUnion(generators, verbose=True)
    return union


def get_feature_dtype() -> Dict:
    dtype = {
        # original
        "userID": "category",
        "testId": "category",
        "assessmentItemID": "category",
        "answerCode": "int8",
        #'Timestamp': 'datetime64[ns]',
        "KnowledgeTag": "category",
        # derived
        "UserAcc": "float16",
        "UserTag2Acc": "float16",
        "UserLastTag1Correct": "category",
        "UserLastTag2Correct": "category",
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
    else:
        raise NotImplementedError
