import numpy as np
from Environment.Parameters import SEED
from Abstractions.ArchitectureBuilder import ArchitectureBuilder
from tensorflow.keras.models import Sequential
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

class TCNModel(ArchitectureBuilder):
    
    def __init__(self, kernel_size = 3, activation='relu', input_dim = None, output_dim=300, max_length = None, emb_matrix = None):
        
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length  = max_length
        self.emb_matrix = emb_matrix
 
        
    def __build_model(self):
        inp = Input( shape=(self.max_length,))
        x = Embedding(input_dim=self.input_dim, 
                    output_dim=self.output_dim, 
                    input_length=self.max_length,
                    # Assign the embedding weight with word2vec embedding marix
                    weights = [self.emb_matrix],
                    # Set the weight to be not trainable (static)
                    trainable = True)(inp)
        
        x = SpatialDropout1D(0.1)(x)
        
        x = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = self.activation, name = 'tcn1')(x)
        x = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = self.activation, name = 'tcn2')(x)
        
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(16, activation="relu")(conc)
        conc = Dropout(0.1)(conc)
        outp = Dense(1, activation="sigmoid")(conc)    

        model = Model(inputs=inp, outputs=outp)
        model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
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
        
    
    
    def get_model(self) -> Sequential:
            model = self.__build_model()
            return model