from sklearn.base import BaseEstimator, TransformerMixin
from pandas import Series
class PreProcessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, name: str):
        self.name = name
    
    
    def fit(self, data: Series):
        print(f'{self.name} started preprocessing !!')
        return self
    
    def transform(self, data: Series):
        return self