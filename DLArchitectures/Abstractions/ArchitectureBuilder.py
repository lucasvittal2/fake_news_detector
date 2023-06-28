from abc import ABC, abstractmethod
from keras.engine.sequential import Sequential

class ArchitectureBuilder(ABC):
    
    def __init__(self):
        
       
        pass
    
    def __build_model(self) -> Sequential:
        pass
    
    def get_model(self) -> Sequential:
        return self.model