from Abstractions.PreProcessor import PreProcessor
import re
from pandas import Series

class EspecialCharRemover(PreProcessor):
    def __init__(self, name='EspcialCharRemover'):
        self.name: str = name
        
    def __remove_especial_chars(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
    def fit(self, data):
        return super().fit(data)
    
    def transform(self, data: Series, y=None):
        print('Removing Especial Characters...')
        print('Especial Characters removed !')
        return data.apply(self.__remove_especial_chars)