import pandas as pd
from sklearn.decomposition import NMF
from features.preprocessor import PreProcessor


class AssessmentLatentFactor(PreProcessor):
    '''문제 - 유저 interaction에서 latent factor를 추출합니다.'''
    
    n_components = 3
    
    def columns(self) -> list[str]:
        return ['userLatentFactor1', 'userLatentFactor2', 'userLatentFactor3', 'assessmentLatentFactor1', 'assessmentLatentFactor2', 'assessmentLatentFactor3']
    
    def create_feature(self, df: pd.DataFrame, **kwargs: dict) -> list[pd.Series]:
        avg_answer_code = df.groupby(['userID', 'assessmentItemID'])['answerCode'].mean().reset_index()
        
        pivot_df = avg_answer_code.pivot(index='userID', columns='assessmentItemID', values='answerCode').fillna(0)
        
        nmf = NMF(n_components=self.n_components)
        
        nmf.fit(pivot_df.T)
        assessment_latent_factor = nmf.transform(pivot_df.T)
        assessment_df = pd.DataFrame(assessment_latent_factor, index=pivot_df.columns)
        merged_df = df.merge(on='assessmentItemID', right=assessment_df)
        ass_latent_series_list = [merged_df.iloc[:, i] for i in range(-self.n_components, 0)]
        
        nmf.fit(pivot_df)
        user_latent_factor = nmf.transform(pivot_df)
        user_df = pd.DataFrame(user_latent_factor, index=pivot_df.index)
        merged_df = df.merge(on='userID', right=user_df)
        user_latent_series_list = [merged_df.iloc[:, i] for i in range(-self.n_components, 0)]
        
        return [*user_latent_series_list, *ass_latent_series_list]