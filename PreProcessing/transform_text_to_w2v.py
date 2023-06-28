import sys
import numpy as np
sys.path.append('../')

from Utils.ProjectPathsSetup import ProjectPathsSetup
from Utils.Concatenator import Cocatenator
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

news_df = Cocatenator().concatenate(fake_news_df, true_news_df)

#instantiate preprocessor

w2vEncoder_text   = Word2VectorEncoder('Word2VectorEncoder', label_type='testing_preprocessing', vec_dim= TEXT_VEC_DIM,window=5,min_count=10, workers=4)
w2vEncoder_title   = Word2VectorEncoder('Word2VectorEncoder', label_type='testing_preprocessing', vec_dim= TITLE_VEC_DIM,window=5,min_count=10, workers=4)

print('\n')
print('Inicializing Preprocessing setup...')
#SET preprocessor for text
preprocessors_text = [
    ('EspecialCharRemover', EspecialCharRemover()),
    ('WordTokeniner', TextTokenizer()),
    ('StopWordsEliminator', StopWordsEliminator(language='english')),
    ('TextRSLPSSteammer', TextRSLPSSteammer()),
    ('Word2VectorEncoder', w2vEncoder_text)
]

#SET preprocessors for title

preprocessors_title = [
    ('EspecialCharRemover', EspecialCharRemover()),
    ('WordTokeniner', TextTokenizer()),
    ('StopWordsEliminator', StopWordsEliminator(language='english')),
    ('TextRSLPSSteammer', TextRSLPSSteammer()),
    ('Word2VectorEncoder', w2vEncoder_title)
]

preprocessors_text = Pipeline(steps = preprocessors_text)
preprocessors_title = Pipeline(steps = preprocessors_title)

print('Setup  done! \n')
print('*'*100)
print('Starting Preprocessing...\n')

print('#'*150)

print('Preprocessing texts data...\n')
text_arrays= preprocessors_text.fit_transform(news_df['text'])


print('Text preprocessed !')
print('#'*150)

print('Preprocessing title data...\n')
title_arrays = preprocessors_title.fit_transform(news_df['title'])
print('#'*150)


print('Preprocessing Done!\n')
print('='*150)


print('Saving Data..\n')
np.savetxt(PREPROCESSED_DATA_PATH + 'w2v_text_arrays.csv', text_arrays, delimiter=',')
np.savetxt(PREPROCESSED_DATA_PATH + 'w2v_title_arrays.csv', title_arrays, delimiter=',')
print('Data Saved')

print('Preprocessing Done ! \n')
print(f'text vocabulary size: {w2vEncoder_text.get_vocab_size()}')
print(f'title vocabulary size: {w2vEncoder_title.get_vocab_size()}')
print('*'*100)