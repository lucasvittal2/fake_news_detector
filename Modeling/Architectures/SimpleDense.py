from keras.engine.sequential import Sequential
from Abstractions.ArchitectureBuilder import ArchitectureBuilder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense


class SimpleDense(ArchitectureBuilder):
    
    def __init__(self, sents_length, input_layer_size= 32,  hidden_layers_size = 16, optimizer='rmsprop'):
        
        self.sents_length = sents_length
        self.input_layer_size = input_layer_size
        self.hidden_layers_size = hidden_layers_size
        self.optimizer = optimizer
        
    def __build_model(self, input_layer_size = 32, hidden_layers_size = 16, optimizer='rmsprop'):
        model = Sequential()
        model.add(Dense(input_layer_size, activation='relu', input_dim= self.sents_length))
        model.add(Dense(hidden_layers_size, activation='relu', input_dim= self.sents_length))
        model.add(Dense(hidden_layers_size, activation='relu', input_dim= self.sents_length))
        model.add(Dense(hidden_layers_size, activation='relu', input_dim= self.sents_length))
        model.add(Dense(hidden_layers_size, activation='relu', input_dim= self.sents_length))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer ,loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def get_building_fuction(self):
        return self.__build_model
    
    def get_model(self) -> Sequential:
        
        input_layer_size = self.input_layer_size
        hidden_layers_size = self.hidden_layers_size
        optimizer = self.optimizer
        model = self.__build_model(input_layer_size , hidden_layers_size , optimizer)
        return model