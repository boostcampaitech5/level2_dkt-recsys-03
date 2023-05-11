from features.preprocessor import PreProcessor
import pandas as pd


class TestType(PreProcessor):
    '''시험지 유형 (00X0000000)'''
    
    def columns(self) -> str:
        return 'testType'
        
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['assessmentItemID'].apply(lambda x : x[2]).astype(int)
    

class TestNum(PreProcessor):
    '''시험지 번호 (0000000XXX)'''
    
    def columns(self) -> str:
        return 'testNum'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['testId'].apply(lambda x: x[-3:]).astype(int)