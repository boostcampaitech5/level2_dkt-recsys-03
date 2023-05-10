import pandas as pd


class PreProcessor():
    '''Feature processor'''
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        '''피쳐를 생성해야 합니다.'''
        pass
    
    def options(self) -> dict:
        return {}
    
    def columns(self) -> str:
        return None