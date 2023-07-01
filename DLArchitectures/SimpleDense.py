from keras.engine.sequential import Sequential
from Abstractions.ArchitectureBuilder import ArchitectureBuilder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense


class SimpleDense(ArchitectureBuilder):
    
    def __init__(self, sents_length):
        
        self.sents_length = sents_length
        
    def __build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim= self.sents_length))
        model.add(Dense(16, activation='relu', input_dim= self.sents_length))
        model.add(Dense(16, activation='relu', input_dim= self.sents_length))
        model.add(Dense(16, activation='relu', input_dim= self.sents_length))
        model.add(Dense(16, activation='relu', input_dim= self.sents_length))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def get_model(self) -> Sequential:
        
        model = self.__build_model()
        return model