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
        "userID": "int",
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
        "UserSolveCnt": "int",
        "RollingTime": "float",
        "UserInteraction1": "category",
        "UserInteraction2": "category",
        "UserInteraction3": "category",
        "UserInteraction4": "category",
        "UserInteraction5": "category",
        "UserTag1Interaction1": "category",
        "UserTag1Interaction2": "category",
        "UserTag1Interaction3": "category",
        "UserTag1Interaction4": "category",
        "UserTag1Interaction5": "category",
        "UserTag2Interaction1": "category",
        "UserTag2Interaction2": "category",
        "UserTag2Interaction3": "category",
        "UserTag2Interaction4": "category",
        "UserTag2Interaction5": "category",
    }
    return dtype


def get_feature_dtype_for_lgbm() -> Dict:
    dtype = {
        # original
        "userID": "int",
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
        "UserTestRetakeCnt": "int",
        "Tag2": "category",
        "UserRecency1": "int",
        "UserRecency2": "int",
        "UserRecency3": "int",
        "UserRecency4": "int",
        "UserRecency5": "int",
        "UserSolveCnt": "int",
        "RollingTime": "float",
        "UserInteraction1": "category",
        "UserInteraction2": "category",
        "UserInteraction3": "category",
        "UserInteraction4": "category",
        "UserInteraction5": "category",
        "UserTag1Interaction1": "category",
        "UserTag1Interaction2": "category",
        "UserTag1Interaction3": "category",
        "UserTag1Interaction4": "category",
        "UserTag1Interaction5": "category",
        "UserTag2Interaction1": "category",
        "UserTag2Interaction2": "category",
        "UserTag2Interaction3": "category",
        "UserTag2Interaction4": "category",
        "UserTag2Interaction5": "category",
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
    elif name == "UserSolveCnt":
        return G.UserSolveCnt()
    elif name == "RollingTime":
        return G.RollingTime()
    elif name == "UserInteraction1":
        return G.UserInteraction1()
    elif name == "UserInteraction2":
        return G.UserInteraction2()
    elif name == "UserInteraction3":
        return G.UserInteraction3()
    elif name == "UserInteraction4":
        return G.UserInteraction4()
    elif name == "UserInteraction5":
        return G.UserInteraction5()
    elif name == "UserTag1Interaction1":
        return G.UserTag1Interaction1()
    elif name == "UserTag1Interaction2":
        return G.UserTag1Interaction2()
    elif name == "UserTag1Interaction3":
        return G.UserTag1Interaction3()
    elif name == "UserTag1Interaction4":
        return G.UserTag1Interaction4()
    elif name == "UserTag1Interaction5":
        return G.UserTag2Interaction5()
    elif name == "UserTag2Interaction1":
        return G.UserTag2Interaction1()
    elif name == "UserTag2Interaction2":
        return G.UserTag2Interaction2()
    elif name == "UserTag2Interaction3":
        return G.UserTag2Interaction3()
    elif name == "UserTag2Interaction4":
        return G.UserTag2Interaction4()
    elif name == "UserTag2Interaction5":
        return G.UserTag2Interaction5()
    else:
        raise NotImplementedError
