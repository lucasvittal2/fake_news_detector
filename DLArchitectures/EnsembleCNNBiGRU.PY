import numpy as np 
from Environment.Parameters import SEED
from keras.engine.sequential import Sequential
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPool1D, Flatten, Dense, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

from Abstractions.ArchitectureBuilder import ArchitectureBuilder

class EnsembleCNNBiGRU(ArchitectureBuilder):
    
    def __init__(self, filters = 100, kernel_size = 3, activation='relu', input_dim = None, output_dim=300, max_length = None, emb_matrix = None):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation= activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.emb_matrix = emb_matrix
        
    
    def __build_model(self) -> Sequential:
 
  
        # Channel 1D CNN
        input1 = Input(shape=(self.max_length,))
        embeddding1 = Embedding(input_dim=self.input_dim, 
                                output_dim=self.output_dim, 
                                input_length=self.max_length, 
                                input_shape=(self.max_length, ),
                                # Assign the embedding weight with word2vec embedding marix
                                weights = [self.emb_matrix],
                                # Set the weight to be not trainable (static)
                                trainable = False)(input1)
        conv1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', 
                    kernel_constraint= MaxNorm( max_value=3, axis=[0,1]))(embeddding1)
        pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
        flat1 = Flatten()(pool1)
        drop1 = Dropout(0.5)(flat1)
        dense1 = Dense(10, activation='relu')(drop1)
        drop1 = Dropout(0.5)(dense1)
        out1 = Dense(1, activation='sigmoid')(drop1)
        
        # Channel BiGRU
        input2 = Input(shape=(self.max_length,))
        embeddding2 = Embedding(input_dim=self.input_dim, 
                                output_dim=self.output_dim, 
                                input_length=self.max_length, 
                                input_shape=(self.max_length, ),
                                # Assign the embedding weight with word2vec embedding marix
                                weights = [self.emb_matrix],
                                # Set the weight to be not trainable (static)
                                trainable = False,
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
        model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def build_pretrained_embedding_matrix(self, word_to_vec_map, word_to_index, emb_mean, emb_std):
        
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
                
        self.embed_matrix = embed_matrix
    
    
    def get_model(self) -> Sequential:
        model = self.__build_model()
        return model