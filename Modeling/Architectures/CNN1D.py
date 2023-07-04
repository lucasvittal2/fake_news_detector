import tensorflow as tf
import numpy as np
from Environment.Parameters import SEED
from keras.engine.sequential import Sequential
from Abstractions.ArchitectureBuilder import ArchitectureBuilder
from tensorflow.keras.constraints import MaxNorm

class CNN1D(ArchitectureBuilder):
    
    def __init__(self, filters = 100, kernel_size = 3, activation='relu', input_dim = None, output_dim=300, max_length = None,  emb_matrix = None ):
        
        self.filters = filters
        self.kernel_size= kernel_size
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.emb_matrix = emb_matrix
    
    def __build_model(self) -> Sequential:
        
        model = tf.keras.models.Sequential([
          tf.keras.layers.Embedding(input_dim=self.input_dim, 
                                  output_dim=self.output_dim, 
                                  input_length= self.max_length,
                                  # Assign the embedding weight 
                                  # with word2vec embedding marix
                                  weights = [self.emb_matrix],
                                  # Set the weight to be not 
                                  # trainable (static)
                                  trainable = True),
        
        tf.keras.layers.Conv1D(filters=self.filters, 
                               kernel_size = self.kernel_size, 
                               activation = self.activation, 
                               # set 'axis' value to the first and 
                               # second axis of conv1D weights 
                               # (rows, cols)
                               kernel_constraint = MaxNorm(
                                   max_value=3, 
                                   axis=[0,1])),
        
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=self.activation, 
                              # set axis to 0 to constrain 
                              # each weight vector of length 
                              # (input_dim,) in dense layer
                              kernel_constraint = MaxNorm(
                                  max_value=3, axis=0)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
    
        model.compile(loss = 'binary_crossentropy', 
                    optimizer = 'adam', 
                    metrics = ['accuracy'])
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
                
        self.emb_matrix = embed_matrix
        return  embed_matrix
    
    def get_model(self) -> Sequential:
       model = self.__build_model()
       return model