from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Embedding
from Abstractions.ArchitectureBuilder import ArchitectureBuilder


class EmbeddingLSTM(ArchitectureBuilder):
    def __init__(self, vo_size:int, embedding_size, lstm_size):
        
        self.lstm_size= lstm_size
        self.embedding_size = embedding_size
        self.vo_size = vo_size
        
    
    def __build_model(self):
        
        model = Sequential()
        model.add(Embedding(self.vo_size, self.embedding_size, name='title_embedding'))
        model.add(LSTM(self.lstm_size))
        model.add( Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        
    def get_model(self):
        model = self.__build_model()
        return model