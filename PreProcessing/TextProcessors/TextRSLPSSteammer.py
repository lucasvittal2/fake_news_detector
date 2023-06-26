import nltk
from pandas import Series

from PreProcessing.Abstractions.PreProcessor import PreProcessor


class TextRSLPSSteammer(PreProcessor):
    
    def __init__(self, name='TextRSLPSSteammer'):
        
        self.name=name
        
        #download nltk steamming package
        
        nltk.download('rslp')
    
    def __steam_text(self, tokens):
        stemmer = nltk.stem.RSLPStemmer()
        return [ stemmer.stem(word) for word in tokens] 
    
    def fit(self, data: Series):
        return super().fit(data)
    
    def transform(self, data: Series):
        print('Steamming words...')
        steammed_words = data.apply(self.__steam_text)
        print('Words Steammed !\n')
        return steammed_words
        
        
        
        