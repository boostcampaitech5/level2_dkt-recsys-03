import numpy as np
import pandas as pd
from feature.modules import FeatureGenerator


class UserAcc(FeatureGenerator):
    """
    유저별 과거 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        past_total = X_.groupby("userID")["answerCode"].cumcount()
        past_correct = X_.groupby("userID")["answerCode"].transform(
            lambda x: x.cumsum().shift()
        )
        acc = past_correct / past_total
        return acc.fillna(0).values.reshape(-1, 1)


class UserItemAcc(FeatureGenerator):
    """
    유저별 현재 풀고 있는 문제의 과거 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        past_total = X_.groupby(["userID", "assessmentItemID"])["answerCode"].cumcount()
        past_correct = X_.groupby(["userID", "assessmentItemID"])[
            "answerCode"
        ].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0).values.reshape(-1, 1)


class UserTag1Acc(FeatureGenerator):
    """
    Tag1: 중분류 태그(KnowledgeTag)
    유저별 현재 풀고 있는 문제의 중분류(KnowledgeTag)에 대한 과거 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["Tag1"] = X_["KnowledgeTag"]
        past_total = X_.groupby(["userID", "Tag1"])["answerCode"].cumcount()
        past_correct = X_.groupby(["userID", "Tag1"])["answerCode"].transform(
            lambda x: x.cumsum().shift()
        )
        acc = past_correct / past_total
        return acc.fillna(0).values.reshape(-1, 1)


class UserTag2Acc(FeatureGenerator):
    """
    Tag2: 대분류 태그(TestId[2])
    유저별 현재 풀고 있는 문제의 대분류(TestId[2])에 대한 과거 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["Tag2"] = X_["testId"].apply(lambda x: x[2]).astype(int)
        past_total = X_.groupby(["userID", "Tag2"])["answerCode"].cumcount()
        past_correct = X_.groupby(["userID", "Tag2"])["answerCode"].transform(
            lambda x: x.cumsum().shift()
        )
        acc = past_correct / past_total
        return acc.fillna(0).values.reshape(-1, 1)


class UserLastTag1Correct(FeatureGenerator):
    """
    * Tag1: 중분류 태그(KnowledgeTag)
    유저별 현재 풀고 있는 문제의 중분류(KnowledgeTag)에 대한 가장 최신 지식상태(정답여부)
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        shift = X_.groupby(["userID", "KnowledgeTag"])["answerCode"].shift()
        return shift.fillna(0).values.reshape(-1, 1)


class UserLastTag2Correct(FeatureGenerator):
    """
    * Tag2: 대분류 태그(TestId[2])
    유저별 현재 풀고 있는 문제의 대분류(TestId[2])에 대한 가장 최신 지식상태(정답여부)
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["Tag2"] = X_["testId"].apply(lambda x: x[2])
        shift = X_.groupby(["userID", "Tag2"])["answerCode"].shift()
        return shift.fillna(0).values.reshape(-1, 1)


class UserTestRollingTime(FeatureGenerator):
    """
    유저별 현재 풀고 있는 시험지에 대한 풀이시간 이동평균
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO
        """
        return


class ItemTestPosi(FeatureGenerator):
    """
    현재 풀고 있는 문항의 시험지 내 위치(문항번호)
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO
        """
        return


class ItemELO(FeatureGenerator):
    """
    TODO
    현재 풀고 있는 문항의 난이도
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO
        """
        return
