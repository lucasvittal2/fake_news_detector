

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
print('Got preprocessing params !!')
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

print("##Instantiate and configuring Algorithms...\n")
print('#EmbeddingDoc\n')

#EmbeddingDoc Encoding

print("Setting up emb_doc_LSTMEMbedding...")
#LSTMEMbedding
emb_doc_lstm_embedding = EmbeddingLSTM(VO_SIZE_TEXT, EMBEDDING_TEXT_SIZE,100).get_model()
print("emb_doc_LSTMEMbedding set up done !!")
print(emb_doc_lstm_embedding.summary())
print('-'*148)

print("Setting up emb_doc_simpleDense...")
#simpleDense
emb_doc_simple_dense = SimpleDense(SENT_LENGTH).get_model()
print("emb_doc_simpleDense set up done !!")
print(emb_doc_simple_dense.summary())
print('-'*148)

#instantiate, setup and get TCN model
print("Setting up emb_doc_TCN...")
emb_doc_tcn = TCNModel(input_dim= vocab_size)
print('Building embedding Matrix...')
emb_doc_tcn.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
emb_doc_tcn = emb_doc_tcn.get_model()
print("emb_doc_TCN set up done !!")
print(emb_doc_tcn.summary())
print('-'*148)


#instantiate, setup and get CNN1D model 
print("Setting up emb_doc_CNN1D...")
emb_doc_cnn1d = CNN1D(input_dim= vocab_size, max_length= SENT_LENGTH)
print('Building embedding Matrix...')
emb_doc_cnn1d.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
emb_doc_cnn1d = emb_doc_cnn1d.get_model()
print("emb_doc_TCN set up done !!")
print(emb_doc_cnn1d.summary())
print('-'*148)

#instantiate, setup and get EnsembleCNNBiGRU modeel
print("Setting up emb_doc_EnsembleCNN...")
emb_doc_ensemblecnn = EnsembleCNNBiGRU(input_dim= vocab_size, max_length=SENT_LENGTH)
print('Building embedding Matrix...')
emb_doc_ensemblecnn.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
emb_doc_ensemblecnn = emb_doc_ensemblecnn.get_model()
print("emb_doc_TCN set up done !!")
print(emb_doc_ensemblecnn.summary())
print('-'*148)


#Word2Vector Enconding
print('#Word2Vector')
#LSTMEMbedding
print("Setting up wv2_LSTMEMbedding...")
w2v_emb_lstm =  EmbeddingLSTM(vocab_params['news_vocab_size'], EMBEDDING__TITLE_SIZE, 100).get_model()
print("wv2_LSTMEMbedding set up done !!")
print(w2v_emb_lstm.summary())
print('-'*148)

#SimpleDense
print("Setting up w2v_simpleDense...")
w2v_simple_dense = SimpleDense(NEWS_VEC_DIM).get_model()
print("w2v_simpleDense set up done !!")
print(w2v_simple_dense.summary())
print('-'*148)

#instantiate and configure TCN
print("Setting up w2v_TCN...")
w2v_tcn = TCNModel(input_dim= vocab_size)
print('Building embedding Matrix...')
w2v_tcn.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
w2v_tcn = w2v_tcn.get_model()
print("w2v_TCN set up done !!")
print(w2v_tcn.summary())
print('-'*148)

#instantiate and configure CNN1D
print("Setting up w2v_CNN1D...")
w2v_cnn1d = CNN1D(input_dim= vocab_size, max_length= NEWS_VEC_DIM)
print('Building embedding Matrix...')
w2v_cnn1d.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
w2v_cnn1d = w2v_cnn1d.get_model()
print("w2v_CNN1D set up done !!")
print(w2v_cnn1d.summary())
print('-'*148)


#instantiate and configure EnsembleCNNBiGRU
print("Setting up w2v_EnsembleCNN...")
w2v_ensemblecnn = EnsembleCNNBiGRU(input_dim= vocab_size, max_length=NEWS_VEC_DIM)
print('Building embedding Matrix...')
w2v_ensemblecnn.build_pretrained_embedding_matrix(google_news_word2vec, word_idxs, gooogle_w2v_emb_mean, gooogle_w2v_emb_std)
print('embedding Matrix built !!')
w2v_ensemblecnn = w2v_ensemblecnn.get_model()
print("w2v_EnsembleCNN set up done !!")
print(w2v_ensemblecnn.summary())
print('-'*148)

#EmbeddingDoc algorithms
emb_doc_algorithms = [  
                
                ( "emb_doc_EmbeddingLSTM", emb_doc_lstm_embedding),
                ( "emb_doc_SimpleDense",  emb_doc_simple_dense),
                ( "emb_doc_TCN",  emb_doc_tcn ),
                ("emb_doc_CNN1D", emb_doc_cnn1d),
                ("emb_doc_EnsembleCNN", emb_doc_ensemblecnn)
                ] 

#Word2Vec algorithms
w2v_algorithms = [                
                
                ("w2v_EmbeddingLSTM", w2v_emb_lstm),
                ("w2v_SimpleDense", w2v_simple_dense),
                ( "w2v_ tcn",  w2v_tcn ),
                ("w2v_CNN1D", w2v_cnn1d),
                ("w2v_EnsembleCNN", w2v_ensemblecnn)
                ]


# splita data  randomly on train and test
news_class = news_df['label']

# split train/test text and title embedded doc
X_emb_doc_train, X_emb_doc_test, y_emb_doc_train , y_emb_doc_test = train_test_split(news_embedding_doc, news_class, train_size=0.8, random_state=SEED)


#split train/test text and title  word vectors

X_w2v_train, X_w2v_test, y_w2v_train , y_w2v_test = train_test_split(news_w2v_encoded, news_class, train_size=0.8, random_state=SEED)


#Train models, save figure and save mean metrics

modelTrainer  = ModelTrainer()





print('#'*148)
#with embeddindDoc Data
print('Training with Embedding encoded Data...\n')
#modelTrainer.train_model(emb_doc_algorithms, X_emb_doc_train, y_emb_doc_train )
print('Training with embedingDoc Encoding data completed !!')

#with w2v Data
print('Training with Embedding encoded Data...\n')
modelTrainer.train_model(w2v_algorithms, X_w2v_train, y_w2v_train )
print('Training with Word2Vector Encoding data completed !!')

#get metrics

sim_metrics = modelTrainer.get_sim_metrics()



print('Saving all metrics obtained during the training...')
jsonHandler.save_json(MODELS_PATH + 'sim_metrics_only_w2v.json', sim_metrics)
print('All metrics Saved !!')
        
print('Modeling Finished Successfully !!!')
        
print('#'*148)
print('='*148)