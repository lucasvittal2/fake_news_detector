import nltk
from nltk.corpus import stopwords
from pandas import Series


from PreProcessing.Abstractions.PreProcessor import PreProcessor



class StopWordsEliminator(PreProcessor):
    
    def __init__(self, name = 'StopWordsEliminator', language='english'):
        
        print("Download nltk 'stopwords' package \n")
        #download stop words
        nltk.download('stopwords')
        
        self.name=name
        self.language = language
        self.stopwords = stopwords.words(language)
   
    def __eliminate_stopwords(self, tokens):
       return [word.lower() for word in tokens if word.lower() not in self.stopwords]
   
        
    def fit(self, data: Series):
        return super().fit(data)
    
    def transform(self, data: Series):
        
        print('Eliminating stopwords...')
        no_stop_words = data.apply(self.__eliminate_stopwords)
        print('Stop words Eliminated !\n')
       
        return no_stop_words
        
        
        