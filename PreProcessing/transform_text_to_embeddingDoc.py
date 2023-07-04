import sys
import numpy as np
sys.path.append('../')

from Utils.ProjectPathsSetup import ProjectPathsSetup
from Utils.Concatenator import Cocatenator
from Utils.JSONHandler import JSONHandler
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

news_df = pd.read_csv(DATASET_PATH  + 'news.csv', sep=',')




print('\n')
print('='*150)
print('Inicializing Preprocessing setup...')
#instantiate preprocessors
especialCharRemover = EspecialCharRemover()
tokenizer = TextTokenizer()
stopwordsEliminator = StopWordsEliminator(language='english',  updtVocab=True)
steammer = TextRSLPSSteammer()
embdocEncoder = EmbeddingDocEncoder(vo_size=500, sent_length=20)
# SET Preprocessors

preprocessors = [
    ('EspecialCharRemover', especialCharRemover),
    ('WordTokeniner', tokenizer),
    ('StopWordsEliminator', stopwordsEliminator),
    ('TextRSLPSSteammer', steammer),
    ('EmbeddingDocEncoder', embdocEncoder)
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
JSONHandler().save_json(PREPROCESSED_DATA_PARAMS_PATH + 'test_get_vocab.json', stopwordsEliminator.word_index)
print('Data Saved')

print('Preprocessing Done ! \n')

print('*'*100)
