from nltk.tokenize import word_tokenize
from PreProcessing.Abstractions.PreProcessor import PreProcessor
from pandas import Series
import nltk

class TextTokenizer(PreProcessor):
    
    def __init__(self, name: str = 'WordTokenizer'):
       
        print("Download \'punkt\' NLTK package\n")
        nltk.download('punkt')
        self.name = name
        print('\n')
    def fit(self, data: Series):
        
        return super().fit(data)
    
    def transform(self, data: Series):
        print("Tokenizing Sentences...")
        tokenized_words = data.apply(nltk.word_tokenize)
        print("Tokenization Done!\n")
       
        return tokenized_words 