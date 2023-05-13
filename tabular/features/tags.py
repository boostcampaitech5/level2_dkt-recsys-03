import pandas as pd
from features.feature_processor import FeatureProcessor


class TagExposureCnt(FeatureProcessor):
    '''태그 노출 횟수'''
    
    def columns(self) -> str:
        return 'tag_exposure_cnt'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        exp_cnt_by_tag = df.groupby('KnowledgeTag').size().reset_index()
        return df.merge(exp_cnt_by_tag, on='KnowledgeTag').iloc[:, -1]