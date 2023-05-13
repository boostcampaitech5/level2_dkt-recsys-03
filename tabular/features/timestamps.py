import pandas as pd
from features.feature_processor import FeatureProcessor


class DayOfWeek(FeatureProcessor):
    '''문제를 풀기 시작한 요일 (0~6) (월~일)'''
    
    def columns(self) -> str:
        return 'day_of_week'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.day_of_week)
    

class Month(FeatureProcessor):
    '''문제를 풀기 시작한 월 (1~12)'''
    
    def columns(self) -> str:
        return 'month'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.month)
    

class Hour(FeatureProcessor):
    '''문제를 풀기 시작한 시'''
    
    def columns(self) -> str:
        return 'hour'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.hour)