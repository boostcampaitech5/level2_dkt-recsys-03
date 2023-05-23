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
        past_correct = X_.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0).values.reshape(-1, 1)


class UserItemAcc(FeatureGenerator):
    """
    유저별 현재 풀고 있는 문제의 과거 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        past_total = X_.groupby(["userID", "assessmentItemID"])["answerCode"].cumcount()
        past_correct = X_.groupby(["userID", "assessmentItemID"])["answerCode"].transform(lambda x: x.cumsum().shift())
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
        past_correct = X_.groupby(["userID", "Tag1"])["answerCode"].transform(lambda x: x.cumsum().shift())
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
        past_correct = X_.groupby(["userID", "Tag2"])["answerCode"].transform(lambda x: x.cumsum().shift())
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


class RollingTime(FeatureGenerator):
    """
    유저별 현재 풀고 있는 시험지에 대한 풀이시간 이동평균(n_rolling=3)
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["solvingtime"] = X_.groupby(["userID", "testId"])["Timestamp"].diff().dt.total_seconds().fillna(0)
        rollingtime = X_.groupby("userID")["solvingtime"].rolling(3).mean()
        return rollingtime.values.reshape(-1, 1)


class ItemNumScaled(FeatureGenerator):
    """
    * 시험지별 min-max scaling 진행
    현재 풀고 있는 문제의 번호
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["ItemNum"] = X_["assessmentItemID"].apply(lambda x: x[-3:]).astype("int8")
        g = X_.groupby("testId")["ItemNum"]
        min_, max_ = g.transform("min"), g.transform("max")
        scaled = (X_["ItemNum"] - min_) / (max_ - min_)
        return scaled.values.reshape(-1, 1)


class Tag1Acc(FeatureGenerator):
    """
    * Tag1:
    현재 풀고 있는 문제의 중분류 태그 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["Tag1"] = X_["KnowledgeTag"]
        agg = X_.groupby("Tag1")["answerCode"].agg(["mean"])
        agg.columns = ["Tag1Acc"]
        X_ = pd.merge(X_, agg, on="Tag1", how="left")
        return X_["Tag1Acc"].values.reshape(-1, 1)


class Tag2Acc(FeatureGenerator):
    """
    * Tag2: 대분류 태그(TestId[2])
    현재 풀고 있는 문제의 대분류 태그 평균 정답률
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["Tag2"] = X_["testId"].apply(lambda x: x[2]).astype("category")
        agg = X_.groupby("Tag2")["answerCode"].agg(["mean"])
        agg.columns = ["Tag2Acc"]
        X_ = pd.merge(X_, agg, on="Tag2", how="left")
        return X_["Tag2Acc"].values.reshape(-1, 1)


class UserTestRetakeCnt(FeatureGenerator):
    """
    유저별 현재 풀고 있는 시험의 재시험 횟수 (0회, 1회, 2회)
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        retake = X_.groupby(["userID", "assessmentItemID"]).cumcount()
        return retake.values.reshape(-1, 1)


class Tag2(FeatureGenerator):
    """
    * Tag2: 대분류 태그(testId[2])
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        tag = X_["testId"].apply(lambda x: x[2])
        return tag.values.reshape(-1, 1)


class UserRecency1(FeatureGenerator):
    """
    유저벌 1시점 전 문제 정답여부
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        recency = X_.groupby("userID")["answerCode"].shift(1)
        return recency.fillna(0).values.reshape(-1, 1)


class UserRecency2(FeatureGenerator):
    """
    유저벌 2시점 전 문제 정답여부
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        recency = X_.groupby("userID")["answerCode"].shift(2)
        return recency.fillna(0).values.reshape(-1, 1)


class UserRecency3(FeatureGenerator):
    """
    유저벌 3시점 전 문제 정답여부
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        recency = X_.groupby("userID")["answerCode"].shift(3)
        return recency.fillna(0).values.reshape(-1, 1)


class UserRecency4(FeatureGenerator):
    """
    유저벌 4시점 전 문제 정답여부
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        recency = X_.groupby("userID")["answerCode"].shift(4)
        return recency.fillna(0).values.reshape(-1, 1)


class UserRecency5(FeatureGenerator):
    """
    유저벌 5시점 전 문제 정답여부
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        recency = X_.groupby("userID")["answerCode"].shift(5)
        return recency.fillna(0).values.reshape(-1, 1)


class UserSolveCnt(FeatureGenerator):
    """
    유저별 총 문제풀이 수
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        agg = pd.DataFrame(X_["userID"].value_counts()).reset_index(drop=False)
        cnt = pd.merge(X_, agg, on="userID", how="left")["count"]
        return cnt.values.reshape(-1, 1)


class UserInteraction1(FeatureGenerator):
    """
    유저별 1시점 전 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["assessmentItemID"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(1).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserInteraction2(FeatureGenerator):
    """
    유저별 2시점 전 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["assessmentItemID"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(2).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserInteraction3(FeatureGenerator):
    """
    유저별 3시점 전 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["assessmentItemID"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(3).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserInteraction4(FeatureGenerator):
    """
    유저별 4시점 전 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["assessmentItemID"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(4).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserInteraction5(FeatureGenerator):
    """
    유저별 5시점 전 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["assessmentItemID"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(5).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag1Interaction1(FeatureGenerator):
    """
    유저별 1시점 전 KnowledgeTag 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["KnowledgeTag"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(1).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag1Interaction2(FeatureGenerator):
    """
    유저별 3시점 전 KnowledgeTag 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["KnowledgeTag"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(2).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag1Interaction3(FeatureGenerator):
    """
    유저별 1시점 전 KnowledgeTag 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["KnowledgeTag"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(3).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag1Interaction4(FeatureGenerator):
    """
    유저별 4시점 전 KnowledgeTag 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["KnowledgeTag"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(4).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag1Interaction5(FeatureGenerator):
    """
    유저별 5시점 전 KnowledgeTag 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["KnowledgeTag"] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(5).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag2Interaction1(FeatureGenerator):
    """
    유저별 1시점 전 testId[2] 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["testId"][2] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(1).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag2Interaction2(FeatureGenerator):
    """
    유저별 2시점 전 testId[2] 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["testId"][2] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(2).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag2Interaction3(FeatureGenerator):
    """
    유저별 3시점 전 testId[2] 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["testId"][2] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(3).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag2Interaction4(FeatureGenerator):
    """
    유저별 4시점 전 testId[2] 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["testId"][2] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(4).fillna("PAD")
        return interaction.values.reshape(-1, 1)


class UserTag2Interaction5(FeatureGenerator):
    """
    유저별 5시점 전 testId[2] 인터렉션
    """

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.copy()
        X_["interaction"] = X_.apply(lambda x: x["testId"][2] + "|" + str(x["answerCode"]), axis=1)
        interaction = X_.groupby("userID")["interaction"].shift(5).fillna("PAD")
        return interaction.values.reshape(-1, 1)
