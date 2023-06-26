import pandas as pd
from pandas import DataFrame


class Cocatenator():
    
    def __init__(self):
        pass
    def __shuffle_data(self,data: DataFrame):
        return data.sample(frac = 1).reset_index(drop = True)
        
    def concatenate(self,df1: DataFrame, df2: DataFrame):
        
        merged_data = pd.concat([df1, df2], axis = 0)
        shuffled_data = self.__shuffle_data(merged_data)
        
        return shuffled_data