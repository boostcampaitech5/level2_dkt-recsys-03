import pandas as pd
from features.feature_processor import FeatureProcessor


class TestType(FeatureProcessor):
    '''시험지 유형 (00X0000000)'''
    
    def columns(self) -> str:
        return 'test_type'
        
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['testId'].apply(lambda x : x[2]).astype(int)
    

class TestNum(FeatureProcessor):
    '''시험지 번호 (0000000XXX)'''
    
    def columns(self) -> str:
        return 'test_num'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['testId'].apply(lambda x: x[-3:]).astype(int)