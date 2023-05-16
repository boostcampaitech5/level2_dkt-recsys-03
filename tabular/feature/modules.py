from sklearn.base import BaseEstimator, TransformerMixin


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self
