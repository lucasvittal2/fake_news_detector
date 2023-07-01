import numpy as np
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm



from  PreProcessing.Abstractions.PreProcessor import PreProcessor


class Word2VectorEncoder(PreProcessor):
    
    
    def __init__(self, name: str, label_type: str, vec_dim,window = 5, min_count=1, workers = 4, train_epochs=50):
        
        self.name = name
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.label_type = label_type
        self.vec_dim = vec_dim
        self.train_epochs = train_epochs
        self.model_w2v = None
        self.corpus = None
        self.vocab_size = None
        
    
    def __labelizeWords(self,words):
        
        print('Labelizeing data...\n')
        label_type = self.label_type
        labelized = []
        LabeledSentence = TaggedDocument
        
        for i,v in tqdm(enumerate(words)):
            
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        print('Labelizing done !\n')
        return labelized
    
    def __build_trainwords(self, tokens):
        print('Building Train Words...\n')
        train_words = [x. words for x in tqdm(tokens)]
        print('Train words ready!\n')
        return  train_words
    
    def __auxBuildWordVector(self, tokens, size, w2v, tfidf):
           
        size= self.vec_dim
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        
        for word in tokens:
            try:
                if type(word) != str:
                    for subword in word:
                        vec += w2v.wv[subword].reshape((1, size)) * tfidf[subword]
                        count += 1.
                else:        
                    vec += w2v.wv[word].reshape((1, size)) * tfidf[word]
                    count += 1.
            except KeyError: # handling the case where the token is not
                            # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec
    
    
    def __buildWordVector(self, labeled_data):
        print('Building scaled vector...\n')
        
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
        matrix = vectorizer.fit_transform([x.words for x in labeled_data])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
       
        train_vecs_w2v = np.concatenate([self.__auxBuildWordVector(z, self.vec_dim, self.model_w2v, tfidf) for z in tqdm(map(lambda x: x.words, labeled_data))])
        train_vecs_w2v = scale(train_vecs_w2v)
        return train_vecs_w2v
    
    def fit(self, data: Series):
        print('Training word2vec model...\n')
        
        #prepare
        corpus= np.array(data.values)
        labelized_corpus = self.__labelizeWords(corpus)
        train_words = self.__build_trainwords(labelized_corpus)

        #train model
        model_w2v = Word2Vec( vector_size= self.vec_dim, window= self.window, min_count= self.min_count, workers= self.workers)
        model_w2v.build_vocab(train_words)
        model_w2v.train(train_words, total_words=model_w2v.corpus_total_words, epochs = self.train_epochs )
        self.model_w2v = model_w2v
        
        #save preprocessed corpous
      
        self.corpus = labelized_corpus
        print('Word2Vector training completed !\n')
        return super().fit(data)
    
    def __shift_and_transform_array(self, words_arrays):
        
        vocab = self.model_w2v.wv.key_to_index
        vo_size= len(list(vocab.keys()))
        vo_gap = 50
        self.vocab_size = vo_size
        return np.array([ np.int64(v + vo_size - vo_gap) for v in words_arrays])
    
    def get_vocab_size(self):
        return self.vocab_size

    
    def transform(self, data: Series):
        
        print('Transforming words in vectors...\n')
        words_as_vectors = self.__buildWordVector(self.corpus)
        words_as_vectors = self.__shift_and_transform_array(words_as_vectors)
        print('\n')
        print('Words Encoded as vectors !')
        
        return words_as_vectors
    
    
    