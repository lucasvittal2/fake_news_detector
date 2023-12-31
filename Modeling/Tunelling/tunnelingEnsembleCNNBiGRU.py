import sys
import json
sys.path.append('../../')

import pandas as pd
import numpy as np



from sklearn.model_selection import  train_test_split
from kerastuner.tuners import RandomSearch


from Environment.Parameters import *
from Environment.PathsParameters import *
from Utils.JSONHandler import JSONHandler
from Utils.ProjectPathsSetup import ProjectPathsSetup
ProjectPathsSetup().add_project_paths('../../')



from Modeling.Tunelling.createEnsembleCNN import *

print('Loading data...')
#instantiate nedd classes 
jsonHandler = JSONHandler()

#load data 
news_df = pd.read_csv(DATASET_PATH  + 'news.csv', sep=',')
news_class = news_df['label']


#get google pretrained w2v
print('Building embedding matrix....\n')

word_idxs = jsonHandler.read_json(PREPROCESSED_DATA_PARAMS_PATH + 'word_indexes.json' )['word_index_news']
vocab_size = len(word_idxs) + 1


print('Embedding Matrix Built !!')
print('Google  Pretrained model setup completed !!')
print('*'*148)

# get feature and label from dataset

print('Getting preprocessed data...')
news_w2v_encoded = np.loadtxt(PREPROCESSED_DATA_PATH +'w2v_word_news_arrays.csv', delimiter= ',')
print('Got preprocessed data !!\n')
print('Splitting data in train and test...')
X_w2v_train, X_w2v_test, y_w2v_train , y_w2v_test = train_test_split(news_w2v_encoded, news_class, train_size=0.8, random_state=SEED)
print('Train and test set defined !!')

#Setup search space
print("*"*148)


print('Data Loaded !!\n')
jsonHandler = JSONHandler()
google_news_word2vec,gooogle_w2v_emb_mean, gooogle_w2v_emb_std =   google_w2v_setup()
word_idxs = jsonHandler.read_json(PREPROCESSED_DATA_PARAMS_PATH + 'word_indexes.json' )['word_index_news']
vocab_size = len(word_idxs) + 1
emb_matrix = build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std) 


print('Starting RadomSearch...')



tuner = RandomSearch(
    lambda hp: create_EnsembleCNN2(hp,vocab_size, emb_matrix),
    objective='val_accuracy',
    max_trials=4,
    executions_per_trial=1,
    directory='/RSData/')

tuner.search_space_summary()


tuner.search([X_w2v_train, X_w2v_train], y_w2v_train,
             epochs=20,
             validation_data=([X_w2v_test, X_w2v_test], y_w2v_test))



tuner.results_summary()



print('RandomSearch were finished successfully !! ')

print('Saving the best simpleDense estimator...')
print('#'*148)
print("Here's your Tunnelled model: \n")
n_best_models = tuner.get_best_models(num_models=1)
print(n_best_models[0].summary())   
print('#'*148)
print('Saving the best simpleDense estimator...')
#jsonHandler.save_json(TUNNELED_MODELS_PATH + 'best_EnsembleCNN.json', str(grid_result.best_params_))
print('Best estimator were saved !!')
print("Tunelling processed Finished Successfully !!")


