from features.preprocessor import PreProcessor
import pandas as pd


class TagExposureCnt(PreProcessor):
    '''태그 노출 횟수'''
    
    def columns(self) -> str:
        return 'tagExposureCnt'
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> pd.Series:
        exp_cnt_by_tag = df.groupby('KnowledgeTag').size().reset_index()
        return df.merge(exp_cnt_by_tag, on='KnowledgeTag').iloc[:, -1]