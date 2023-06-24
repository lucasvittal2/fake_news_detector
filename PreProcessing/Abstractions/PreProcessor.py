from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, name):
        self.name = name
    
    
    def fit(self, data):
        print(f'{self.name} started preprocessing !!')
        return self
    
    def transform(self, data, column):
        return self