from pandas import Series
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from PreProcessing.Abstractions.PreProcessor import PreProcessor


class EmbeddingDocEncoder(PreProcessor):
    
    def __init__(self, vo_size, sent_length ,name="EmbeddingDocEncoder"):
        
        self.name = name
        self.vo_size = vo_size
        self.sent_length = sent_length
        self.sentences = None
        
    
    def fit(self, data: Series):
        
        print('Getting Setences...\n')
        self.sentences  = [' '.join(words) for words in data]
        print('Got Senteces !\n')
        return super().fit(data)
    
    def get_vocab_size(self):
        return self.vo_size
    
    def transform(self, data: Series):
        print('Building onehotrep encoding..\n')
        onehot_rep = [one_hot(words, self.vo_size) for words in self.sentences]
        print('onethotrep encoding built !\n')
        print('Building Embedding doc...\n')
        embedded_doc = pad_sequences(onehot_rep, padding='pre', maxlen=self.sent_length)
        print('Embedding Doc Built !\n')
        return embedded_doc