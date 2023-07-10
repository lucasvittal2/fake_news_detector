import numpy as np 
from Environment.Parameters import *
from Utils.JSONHandler import JSONHandler
from keras.engine.sequential import Sequential
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPool1D, Flatten, Dense, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from Environment.PathsParameters import *
from gensim.models import KeyedVectors


def build_pretrained_embedding_matrix( word_to_vec_map, word_to_index, emb_mean, emb_std):
    
    np.random.seed(SEED)
    
    # adding 1 to fit Keras embedding (requirement)
    vocab_size = len(word_to_index) + 1
    # define dimensionality of your pre-trained word vectors
    emb_dim = word_to_vec_map.word_vec('handsome').shape[0]
    
    # initialize the matrix with generic normal distribution
    embed_matrix = np.random.normal(emb_mean, 
                                    emb_std, 
                                    (vocab_size, emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th 
    # word of the vocabulary
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            
            embed_matrix[idx] = word_to_vec_map.get_vector(word)
            
   
    return  embed_matrix

def create_EnsembleCNN(input_dim, filters, kernel_size, emb_matrix,activation, optimizer, trainable ) -> Sequential:
     
  
        # Channel 1D CNN
        
        input1 = Input(shape=(NEWS_VEC_DIM,))
        embeddding1 = Embedding(input_dim=input_dim, 
                                output_dim=GOOGLE_VEC_DIM, 
                                input_length=NEWS_VEC_DIM,
                                # Assign the embedding weight with word2vec embedding marix
                                weights = [emb_matrix],
                                # Set the weight to be not trainable (static)
                                trainable = trainable)(input1)
        conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, 
                    kernel_constraint= MaxNorm( max_value=3, axis=[0,1]))(embeddding1)
        pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
        flat1 = Flatten()(pool1)
        drop1 = Dropout(0.5)(flat1)
        dense1 = Dense(10, activation=activation)(drop1)
        drop1 = Dropout(0.5)(dense1)
        out1 = Dense(1, activation='sigmoid')(drop1)
        
        # Channel BiGRU
        input2 = Input(shape=(NEWS_VEC_DIM,))
        embeddding2 = Embedding(input_dim=input_dim, 
                                output_dim=GOOGLE_VEC_DIM, 
                                input_length=NEWS_VEC_DIM,
                                # Assign the embedding weight with word2vec embedding marix
                                weights = [emb_matrix],
                                # Set the weight to be not trainable (static)
                                trainable = trainable,
                                mask_zero=True)(input2)
        gru2 = Bidirectional(GRU(64))(embeddding2)
        drop2 = Dropout(0.5)(gru2)
        out2 = Dense(1, activation='sigmoid')(drop2)
        
        # Merge
        merged = concatenate([out1, out2])
        
        # Interpretation
        outputs = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[input1, input2], outputs=outputs)
        
        # Compile
        model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
def google_w2v_setup():
    print('*'*148)
    print('Getting Google Pretrained Model')
    google_news_word2vec = KeyedVectors.load_word2vec_format(
                GOOGLE_PRETRAINED_MODEL_PATH, 
                binary=True)

    gooogle_w2v_emb_mean = google_news_word2vec.vectors.mean()
    gooogle_w2v_emb_std = google_news_word2vec.vectors.std()
    print('Got Google Pretrained Model !!')
    print('*'*148)
    
    return google_news_word2vec, gooogle_w2v_emb_mean, gooogle_w2v_emb_std
    
    
def create_EnsembleCNN2(hp,vocab_size, emb_matrix) -> Sequential:
    
   
    
    
    hp_trainable = hp.Boolean([ True, False])
    hp_kernel_size = hp.Int('kernel_size', min_value = 3, max_value = 11, step= 2)
    hp_filters = hp.Int('filters', min_value = 100, max_value = 300, step= 50)
    hp_activation = hp.Choice('activation',values=['relu','sigmoid','tanh','selu','elu','softmax'])
    hp_batch_size = hp.Choice('batch_size', values=[10, 20, 40, 60, 80, 100, 128, 512]) 
    hp_epochs = hp.Int('epoch', min_value = 10, max_value = 100, step= 10)
    hp_optimizers =hp.Choice('optimizers', values= ['sgd', 'adam','adagrad','adamax','adadelta','rmsprop','nadam','ftrl'])
   
   
    
    # Channel 1D CNN
    
    input1 = Input(shape=(NEWS_VEC_DIM,))
    embeddding1 = Embedding(input_dim=vocab_size, 
                            output_dim=GOOGLE_VEC_DIM, 
                            input_length=NEWS_VEC_DIM,
                            # Assign the embedding weight with word2vec embedding marix
                            weights = [emb_matrix],
                            # Set the weight to be not trainable (static)
                            trainable = hp_trainable)(input1)
    conv1 = Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation=hp_activation, 
                kernel_constraint= MaxNorm( max_value=3, axis=[0,1]))(embeddding1)
    pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
    flat1 = Flatten()(pool1)
    drop1 = Dropout(0.5)(flat1)
    dense1 = Dense(10, activation=hp_activation)(drop1)
    drop1 = Dropout(0.5)(dense1)
    out1 = Dense(1, activation='sigmoid')(drop1)
    
    # Channel BiGRU
    input2 = Input(shape=(NEWS_VEC_DIM,))
    embeddding2 = Embedding(input_dim=vocab_size, 
                            output_dim=GOOGLE_VEC_DIM, 
                            input_length=NEWS_VEC_DIM,
                            # Assign the embedding weight with word2vec embedding marix
                            weights = [emb_matrix],
                            # Set the weight to be not trainable (static)
                            trainable = hp_trainable,
                            mask_zero=True)(input2)
    gru2 = Bidirectional(GRU(64))(embeddding2)
    drop2 = Dropout(0.5)(gru2)
    out2 = Dense(1, activation='sigmoid')(drop2)
    
    # Merge
    merged = concatenate([out1, out2])
    
    # Interpretation
    outputs = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[input1, input2], outputs=outputs)
    
    # Compile
    model.compile( loss='binary_crossentropy', optimizer=hp_optimizers, metrics=['accuracy'])
    return model