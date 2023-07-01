import sys
import numpy as np
sys.path.append('../')

from Utils.ProjectPathsSetup import ProjectPathsSetup
from Utils.Concatenator import Cocatenator
ProjectPathsSetup().add_project_paths('../')


from Environment.PathsParameters import *
from sklearn.pipeline import Pipeline
import pandas as pd




from PreProcessing.TextProcessors.EspecialCharRemover import EspecialCharRemover
from PreProcessing.TextProcessors.TextTokenizer import TextTokenizer
from PreProcessing.TextProcessors.StopWordsEliminator import StopWordsEliminator
from PreProcessing.TextProcessors.TextRSLPSSteammer import TextRSLPSSteammer
from PreProcessing.Encoders.EmbeddingDocEncoder import EmbeddingDocEncoder

# load data
fake_news_df = pd.read_csv(TRUE_NEWS_DATASET)
true_news_df = pd.read_csv(FAKE_NEWS_DATASET)

news_df = Cocatenator().concatenate( fake_news_df, true_news_df)

#build news data
news_df['news']  = news_df[['title', 'text']].apply(lambda row: row['title'] + ' '  + row['text'], axis = 1) #joint title with text


print('\n')
print('='*150)
print('Inicializing Preprocessing setup...')

# SET Preprocessors

preprocessors = [
    ('EspecialCharRemover', EspecialCharRemover()),
    ('WordTokeniner', TextTokenizer()),
    ('StopWordsEliminator', StopWordsEliminator(language='english')),
    ('TextRSLPSSteammer', TextRSLPSSteammer()),
    ('EmbeddingDocEncoder', EmbeddingDocEncoder(vo_size=500, sent_length=20))
]
print('Setup  done! \n')
print('*'*100)
print('Starting Preprocessing News...\n')

preprocessors_pipeline= Pipeline(steps = preprocessors)

# Execute preprocessing 

word_news_arrays= preprocessors_pipeline.fit_transform(news_df['news'])


print('News Preprocessing Done!\n')
print('='*150)

#save data 

print('Saving Data..\n')
np.savetxt(PREPROCESSED_DATA_PATH + 'embedding_doc_word_news_arrays.csv', word_news_arrays, delimiter=',')

print('Data Saved')

print('Preprocessing Done ! \n')

print('*'*100)
