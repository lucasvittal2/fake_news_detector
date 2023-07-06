import sys
import json
sys.path.append('../../')

import pandas as pd
import numpy as np


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from gensim.models import KeyedVectors

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
print('*'*148)
print('Getting Google Pretrained Model')
google_news_word2vec = KeyedVectors.load_word2vec_format(
            GOOGLE_PRETRAINED_MODEL_PATH, 
            binary=True)

gooogle_w2v_emb_mean = google_news_word2vec.vectors.mean()
gooogle_w2v_emb_std = google_news_word2vec.vectors.std()
print('Got Google Pretrained Model !!')
print('*'*148) 
print('Building embedding matrix....\n')

word_idxs = jsonHandler.read_json(PREPROCESSED_DATA_PARAMS_PATH + 'word_indexes.json' )['word_index_news']
vocab_size = len(word_idxs) + 1


print('Embedding Matrix Built !!')
print('Google  Pretrained model setup completed !!')
print('*'*148)


#get model creator



# get feature and label from dataset

print('Getting preprocessed data...')
news_w2v_encoded = np.loadtxt(PREPROCESSED_DATA_PATH +'w2v_word_news_arrays.csv', delimiter= ',')
print('Got preprocessed data !!\n')
print('Splitting data in train and test...')
X_w2v_train, X_w2v_test, y_w2v_train , y_w2v_test = train_test_split(news_w2v_encoded, news_class, train_size=0.8, random_state=SEED)
print('Train and test set defined !!')

#Setup search space
print("*"*148)
print('Getting hyperparameters...')
input_dim = [vocab_size]
trainable = [ True, False]
kernel_size = [3, 5, 7, 9, 11]
filters = [100,150,200,250,300]
activation_functions =['relu','sigmoid','tanh','selu','elu','softmax']
batch_size = [10, 20, 40, 60, 80, 100, 128, 512]
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
optimizers = [ 'adam','adafactor','adamw','adagrad','adamax','adadelta','rmsprop','sgd','nadam','ftrl']
embedding_matrix = [ build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std) ] 

param_grid = dict( 
                   model__input_dim = input_dim,
                   model__trainable = trainable,
                   model__kernel_size = kernel_size,
                   model__filters=filters,
                   model__activation = activation_functions,
                   model__emb_matrix = embedding_matrix,
                   model__optimizer =  optimizers,
                   batch_size=batch_size,
                   epochs=epochs
                   
                )

print('Hyperparameter set defined !!')
print("*"*148)


print('Data Loaded !!\n')


# create Keras model wrapped by SkLearn.KerasClassifier
shape = np.array(X_w2v_train).shape
model = KerasClassifier(model = create_EnsembleCNN, epochs=100, batch_size=10, verbose=10)



print('Starting RadomSearch...')

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=3, cv=3, verbose=10, scoring='accuracy')
grid_result = grid.fit([X_w2v_train, X_w2v_train], y_w2v_train, workers= 3, verbose=10)

print('RandomSearch were finished successfully !! ')

print('Saving the best simpleDense estimator...')
print('#'*148)
print("Here's your Tunnelled model: \n")
print(grid_result.best_params_)
print(grid_result.best_estimator_)
print('#'*148)
print('Saving the best simpleDense estimator...')
jsonHandler.save_json(TUNNELED_MODELS_PATH + 'best_EnsembleCNN.json', str(grid_result.best_params_))
print('Best estimator were saved !!')
print("Tunelling processed Finished Successfully !!")


