import pandas as pd
from features.feature_processor import FeatureProcessor

class RollingTime(FeatureProcessor):
    '''과거 문제 풀이시간 이동 평균'''
    
    def columns(self) -> str:
        return 'rolling_time'
    
    
    def options(self) -> dict:
        return {
            'n_rolling': [3, 6, 9, 12],
        }
    
    def create_feature(self, 
                       df: pd.DataFrame, 
                       n_rolling: int,
                       **kwargs: dict) -> pd.Series:
        
        ### get diff
        selected = df[['Timestamp','userID','testId']]
        diff = selected.groupby(['userID','testId']).diff()
            
        ### get solving_time
        solving_time = diff['Timestamp'].apply(lambda x: x.total_seconds() if pd.notna(x) else 0)
        solving_time = pd.concat([solving_time, df['userID']], ignore_index=False, axis=1)

        ## get rolling_time
        rolling_time = pd.DataFrame(solving_time.groupby(['userID'])['Timestamp'].rolling(n_rolling).mean().values)
        
        return rolling_time