from features.preprocessor import PreProcessor
import pandas as pd


class SolvingTime(PreProcessor):
    '''문제 풀이시간'''
    
    def columns(self) -> str:
        return 'solvingTime'
    
    
    def options(self) -> dict:
        return {
            'contain_repeat_cnt': [False, True],
            'fill_nan': [0],
            'type': ['second', 'minute']
        }
    
    def create_feature(self, df: pd.DataFrame, 
                       contain_repeat_cnt: bool,
                       fill_nan: int,
                       type: str,
                       **kwargs: dict) -> pd.Series:
        ### get diff
        if contain_repeat_cnt:
            selected = df[['Timestamp','userID','testId','repeatCnt']]
            diff = selected.groupby(['userID','testId','repeatCnt']).diff()
        else:
            selected = df[['Timestamp','userID','testId']]
            diff = selected.groupby(['userID','testId']).diff()
            
        ### get solving_time
        solving_time = diff.shift(-1)['Timestamp'].apply(lambda x: x.total_seconds() if pd.notna(x) else fill_nan)
        
        if type == 'second':
            pass
        elif type == 'minute':
            solving_time = solving_time.apply(lambda x: x // 60)
            
        return solving_time