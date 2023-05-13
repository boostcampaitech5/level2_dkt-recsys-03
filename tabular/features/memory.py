import pandas as pd
from features.feature_processor import FeatureProcessor


class UserAcc(FeatureProcessor):
    """
    유저별 과거 평균 정답률
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_acc'
    
    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        past_total = df.groupby('userID')['answerCode'].cumcount()
        past_correct = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0)


class UserItemAcc(FeatureProcessor):
    """
    유저별 현재 풀고 있는 문제의 과거 평균 정답률
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_item_acc'

    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        past_total = df.groupby(['userID', 'assessmentItemID'])['answerCode'].cumcount()
        past_correct = df.groupby(['userID', 'assessmentItemID'])['answerCode'].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0)


class UserTag1Acc(FeatureProcessor):
    """
    Tag1: 중분류 태그(KnowledgeTag)
    유저별 현재 풀고 있는 문제의 중분류(KnowledgeTag)에 대한 과거 평균 정답률
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_tag1_acc'
    
    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        df['Tag1'] = df['KnowledgeTag']
        past_total = df.groupby(['userID', 'Tag1'])['answerCode'].cumcount()
        past_correct = df.groupby(['userID', 'Tag1'])['answerCode'].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0)


class UserTag2Acc(FeatureProcessor):
    """
    Tag2: 대분류 태그(TestId[2])
    유저별 현재 풀고 있는 문제의 대분류(TestId[2])에 대한 과거 평균 정답률
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_tag2_acc'

    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        df['Tag2'] = df['testId'].apply(lambda x : x[2]).astype(int)
        past_total = df.groupby(['userID', 'Tag2'])['answerCode'].cumcount()
        past_correct = df.groupby(['userID', 'Tag2'])['answerCode'].transform(lambda x: x.cumsum().shift())
        acc = past_correct / past_total
        return acc.fillna(0)


class UserLastTag1Correct(FeatureProcessor):
    """
    * Tag1: 중분류 태그(KnowledgeTag)
    유저별 현재 풀고 있는 문제의 중분류(KnowledgeTag)에 대한 가장 최신 지식상태(정답여부) 
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_last_tag1_correct'

    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        df['Tag1'] = df['KnowledgeTag']
        shift = df.groupby(['userID', 'Tag1'])['answerCode'].shift()
        return shift.fillna(0)


class UserLastTag2Correct(FeatureProcessor):
    """
    * Tag2: 대분류 태그(TestId[2])
    유저별 현재 풀고 있는 문제의 대분류(TestId[2])에 대한 가장 최신 지식상태(정답여부)
    결측값 있음
    """
    def columns(self) -> str:
        return 'user_last_tag2_correct'

    def create_feature(self, df: pd.DataFrame) -> pd.Series:
        df['Tag2'] = df['testId'].apply(lambda x : x[2]).astype(int)
        shift = df.groupby(['userID', 'Tag2'])['answerCode'].shift().astype("category")
        return shift.fillna(0)