import sys
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


fake_news_df = pd.read_csv(TRUE_NEWS_DATASET)
true_news_df = pd.read_csv(FAKE_NEWS_DATASET)

news_df = Cocatenator().concatenate( fake_news_df, true_news_df)

print('\n')
print('Inicializing Preprocessing setup...')
preprocessors = [
    ('EspecialCharRemover', EspecialCharRemover()),
    ('WordTokeniner', TextTokenizer()),
    ('StopWordsEliminator', StopWordsEliminator(language='english')),
    ('TextRSLPSSteammer', TextRSLPSSteammer()),
    ('EmbeddingDocEncoder', EmbeddingDocEncoder(vo_size=500, sent_length=20))
]
print('Setup  done! \n')
print('*'*100)
print('Starting Preprocessing...\n')

preprocessors = Pipeline(steps = preprocessors)
arrays= preprocessors.fit_transform(news_df['text'])

print(arrays)

print('Preprocessing Done ! \n')
print('*'*100)
