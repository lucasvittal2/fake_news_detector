from PreProcessing.Abstractions.PreProcessor import PreProcessor
import re
from pandas import Series

class EspecialCharRemover(PreProcessor):
    def __init__(self, name='EspcialCharRemover'):
        self.name: str = name
        
    def __remove_especial_chars(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
    def fit(self, data: Series):
        return super().fit(data)
    
    def transform(self, data: Series, y=None):
        print('Removing Especial Characters...')
        no_esp_char = data.apply(self.__remove_especial_chars)
        print('Especial Characters removed !\n')
        
        return no_esp_char