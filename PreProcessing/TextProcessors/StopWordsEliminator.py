import nltk
from nltk.corpus import stopwords
from pandas import Series


from PreProcessing.Abstractions.PreProcessor import PreProcessor
from keras.preprocessing.text import Tokenizer as KerasTokenizer


class StopWordsEliminator(PreProcessor):
    
    def __init__(self, name = 'StopWordsEliminator', language='english', updtVocab=False):
        
        print("Download nltk 'stopwords' package \n")
        #download stop words
        nltk.download('stopwords')
        
        self.name=name
        self.language = language
        self.stopwords = stopwords.words(language)
        self.updtVocab = updtVocab
        self.word_index = []
   
    def __eliminate_stopwords(self, tokens):
       no_stop_words =   [word.lower() for word in tokens if word.lower() not in self.stopwords]
       return [word for word in no_stop_words if len(word) > 1]
   
    def __update_vocab(self, sentences):
        kerasTokenizer = KerasTokenizer()
        kerasTokenizer.fit_on_texts(sentences)
        self.word_index = kerasTokenizer.word_index
    '''  for tokens in sentences:
            actual_vocab_size = len(self.word_index)
            for idx, token in enumerate(tokens):
                if token not in [word_idx[1] for word_idx in self.word_index ]:
                    self.word_index.append( (idx + actual_vocab_size, token) ) '''
        
    def fit(self, data: Series):
        return super().fit(data)
    
    def transform(self, data: Series):
        
        print('Eliminating stopwords...')
        no_stop_words = data.apply(self.__eliminate_stopwords)
        if self.updtVocab:
            self.__update_vocab(no_stop_words)
        print('Stop words Eliminated !\n')
       
        return no_stop_words
        
        
        