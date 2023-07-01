import json
import pandas as pd
from Environment.PathsParameters import PREPROCESSED_DATA_PATH

class JSONHandler():
    
    def __init__(self):
        pass
    
    def read_json(self,file_path):
        with open(file_path, 'r') as  file:
            data = json.load(file)   
        return data
    
    
    def save_json(self, file_path, data):
        with open(file_path, 'w') as  file:
            json.dump(data, file)
            
            
            