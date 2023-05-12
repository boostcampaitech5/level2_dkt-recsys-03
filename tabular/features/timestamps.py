from features.preprocessor import PreProcessor
import pandas as pd


class DayOfWeek(PreProcessor):
    '''문제를 풀기 시작한 요일 (0~6) (월~일)'''
    
    def columns(self) -> str:
        return 'dayOfWeek'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.day_of_week)
    

class Month(PreProcessor):
    '''문제를 풀기 시작한 월 (1~12)'''
    
    def columns(self) -> str:
        return 'month'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.month)
    

class Hour(PreProcessor):
    '''문제를 풀기 시작한 시'''
    
    def columns(self) -> str:
        return 'hour'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        return df['Timestamp'].apply(lambda x: x.hour)