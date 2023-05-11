import pandas as pd
from typing import Union


class PreProcessor():
    '''Feature processor'''
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> Union[pd.Series, list[pd.Series]]:
        '''피쳐를 생성해야 합니다.'''
        pass
    
    def options(self) -> dict:
        return {}
    
    def columns(self) -> Union[str, list[str]]:
        return None