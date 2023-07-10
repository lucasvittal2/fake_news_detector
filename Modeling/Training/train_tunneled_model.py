

import numpy as np
import pickle
import pandas as pd
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

sys.path.append('../../')


from Utils.ProjectPathsSetup import ProjectPathsSetup
ProjectPathsSetup().add_project_paths('../')
from Utils.JSONHandler import JSONHandler


from Modeling.Architectures.SimpleDense import SimpleDense
from Modeling.Architectures.EmbeddingLSTM import EmbeddingLSTM
from Modeling.Architectures.TCNModel import TCNModel
from Modeling.Architectures.EnsembleCNNBiGRU import EnsembleCNNBiGRU
from Modeling.Architectures.CNN1D import CNN1D
from Modeling.Training.ModelTrainer import ModelTrainer
from Utils.GraphicPlotter import GraphicPlotter

from Environment.PathsParameters import *
from Environment.Parameters import *






    
    
print('='*148)
print('*'*148)    
print('#Loading data...\n')

print('Loading. Dataset...')    
#load data 
news_df = pd.read_csv(DATASET_PATH  + 'news.csv', sep=',')
print('Got Dataset !!')    

print('Loading PreProcessed News Content...')
#embedding doc
news_embedding_doc = np.loadtxt(PREPROCESSED_DATA_PATH +'embedding_doc_word_news_arrays.csv', delimiter= ',')


#word2vec
news_w2v_encoded = np.loadtxt(PREPROCESSED_DATA_PATH +'w2v_word_news_arrays.csv', delimiter= ',')
print('Got PreProcessed News !!')
    
#Instantiate necessary classes

grapPlotter = GraphicPlotter()
jsonHandler = JSONHandler()
    
print('*'*148) 
print('#Making the modeling setup...')
# Set Seed to guarantee reprodubility
np.random.seed(SEED)
tf.random.set_seed(SEED)



print('Loading Preprocesing Parms...')

#load Preprocessing Parms
vocab_params = jsonHandler.read_json(PREPROCESSED_DATA_PARAMS_PATH + 'vocab_params.json' )
word_idxs = jsonHandler.read_json(PREPROCESSED_DATA_PARAMS_PATH + 'word_indexes.json' )['word_index_news']
vocab_size = len(word_idxs) + 1
print('Got preprocessing params !!\n')
print('*'*148)
print('Loading Google Pretrained Model...')

#load google pretrained model and make necessary setup 
google_news_word2vec = KeyedVectors.load_word2vec_format(
            GOOGLE_PRETRAINED_MODEL_PATH, 
            binary=True)

gooogle_w2v_emb_mean = google_news_word2vec.vectors.mean()
gooogle_w2v_emb_std = google_news_word2vec.vectors.std()
print('Got Google Pretrained Model !!')
print('*'*148) 

#Instantiate algorithm

print("##Instantiate and configuring Algorithms EnsembleCNNBiGRU with best parameters found\n")

ensemble_cnn_best_parms = {
    "trainable": False,
    "kernel_size": 11,
    "filters": 150,
    "activation": 'tanh',
    "optimizer": 'adamax',
    
}
#EmbeddingDoc Encoding




#Word2Vector Enconding



#instantiate and configure EnsembleCNNBiGRU
print("Setting up w2v_EnsembleCNN...")
w2v_ensemblecnn = EnsembleCNNBiGRU(input_dim= vocab_size, max_length=NEWS_VEC_DIM, trainable=False, kernel_size=7, filters=200, activation='tanh', optimizer='adamax' )
print('Building embedding Matrix...')
w2v_ensemblecnn.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
w2v_ensemblecnn = w2v_ensemblecnn.get_model()
print("w2v_EnsembleCNN set up done !!")
print(w2v_ensemblecnn.summary())
print('-'*148)

#EmbeddingDoc algorithms


#Word2Vec algorithms
tunelled_model = [                
                    ("w2v_EnsembleCNN_Tunelled2", w2v_ensemblecnn)
                ]


# splita data  randomly on train and test
news_class = news_df['label']



print('Splitting data in train/test sets...\n')
#split train/test text and title  word vectors

X_w2v_train, X_w2v_test, y_w2v_train , y_w2v_test = train_test_split(news_w2v_encoded, news_class, train_size=0.8, random_state=SEED)
print('Data Ready for training !!\n')

#Train models, save figure and save mean metrics

modelTrainer  = ModelTrainer()





print('#'*148)

#with w2v Data
print('Training the Best model found...\n')
modelTrainer.train_model(tunelled_model, X_w2v_train, y_w2v_train )
print('Training completed !!')

#get metrics

sim_metrics = modelTrainer.get_sim_metrics()



print('Saving all metrics obtained during the training...')
jsonHandler.save_json(MODELS_PATH + 'sim_metrics_EnsembleCNN_tunneld2.json', sim_metrics)
print('All metrics Saved !!')
        
print('Modeling Finished Successfully !!!')
        
print('#'*148)
print('='*148)