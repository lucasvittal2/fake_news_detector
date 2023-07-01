import sys
import json
import numpy as np
sys.path.append('../')

from Utils.ProjectPathsSetup import ProjectPathsSetup
from Utils.Concatenator import Cocatenator
from Utils.JSONHandler import JSONHandler
ProjectPathsSetup().add_project_paths('../')

from Environment.Parameters import *
from Environment.PathsParameters import *
from sklearn.pipeline import Pipeline
import pandas as pd




from PreProcessing.TextProcessors.EspecialCharRemover import EspecialCharRemover
from PreProcessing.TextProcessors.TextTokenizer import TextTokenizer
from PreProcessing.TextProcessors.StopWordsEliminator import StopWordsEliminator
from PreProcessing.TextProcessors.TextRSLPSSteammer import TextRSLPSSteammer
from PreProcessing.Encoders.Word2VectorEncoder import Word2VectorEncoder


fake_news_df = pd.read_csv(TRUE_NEWS_DATASET)
true_news_df = pd.read_csv(FAKE_NEWS_DATASET)


#to write json files

jsonHandler = JSONHandler()

news_df = Cocatenator().concatenate(fake_news_df, true_news_df)
news_df['news']  = news_df[['title', 'text']].apply(lambda row: row['title'] + ' '  + row['text'], axis = 1) #joint title with text

#instantiate preprocessors

charRemover= EspecialCharRemover()
tokenizer = TextTokenizer()
stopWordsEliminator= StopWordsEliminator(language='english')
stemmer= TextRSLPSSteammer()
w2vEncoder  = Word2VectorEncoder('Word2VectorEncoder', label_type='testing_preprocessing', vec_dim= TITLE_VEC_DIM,window=5,min_count=10, workers=4)


print('\n')
print('Inicializing Preprocessing setup...')

#SET preprocessors 

preprocessors = [
    ('EspecialCharRemover', charRemover),
    ('WordTokeniner', tokenizer),
    ('StopWordsEliminator', stopWordsEliminator),
    ('TextRSLPSSteammer',stemmer),
    ('Word2VectorEncoder', w2vEncoder)
]

preprocessors_pipeline = Pipeline(steps = preprocessors)


print('Setup  done! \n')
print('*'*100)
print('Starting Preprocessing...\n')

print('#'*150)

print('Preprocessing news data...\n')
word_news_arrays= preprocessors_pipeline.fit_transform(news_df['text'])


print('News preprocessed !')
print('#'*150)

print('Preprocessing Done!\n')
print('='*150)


print('Saving Data..\n')
news_vocab_size = w2vEncoder.get_vocab_size()
np.savetxt(PREPROCESSED_DATA_PATH + 'w2v_word_news_arrays.csv', word_news_arrays, delimiter=',')


jsonHandler.save_json(PREPROCESSED_DATA_PARAMS_PATH + 'vocab_params.json', 
                      {"news_vocab_size": news_vocab_size}
                      )

jsonHandler.save_json(PREPROCESSED_DATA_PARAMS_PATH + 'word_indexes.json',{'word_index_news': stopWordsEliminator.word_index}  )

print('Data Saved')

print('Preprocessing Done ! \n')
print(f'News vocabulary size: {news_vocab_size}')
print('*'*100)