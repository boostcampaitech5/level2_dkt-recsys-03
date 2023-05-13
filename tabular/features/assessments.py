import pandas as pd
from features.feature_processor import FeatureProcessor


class AssessmentNum(FeatureProcessor):
    '''문제 번호 (0000000XXX)'''
    
    def columns(self) -> str:
        return 'assessment_num'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['assessmentItemID'].apply(lambda x: x[-3:]).astype(int)
    

class UserSolvedCnt(FeatureProcessor):
    '''유저별 문제 풀이 횟수'''
    
    def columns(self) -> str:
        return 'user_solved_cnt'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        slv_cnt_by_user = df.groupby('userID').size().reset_index()
        return df.merge(slv_cnt_by_user, on='userID').iloc[:, -1]


class RepeatCnt(FeatureProcessor):
    '''문제 풀이 반복 횟수 (복습 확인용)'''
    
    def columns(self) -> str:
        return 'repeat_cnt'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df.groupby(['userID', 'assessmentItemID']).cumcount()