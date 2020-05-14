from gensim.models.word2vec import Word2Vec
import numpy as np
from utils.config import embedding_matrix_path
import codecs
import logging

def load_vocab(word2vec_model):
    model = Word2Vec.load(word2vec_model)

    vocab_to_index = {token: index for index, token in enumerate(model.wv.index2word)}
    index_to_vocab = {index: token for index, token in enumerate(model.wv.index2word)}
    return vocab_to_index, index_to_vocab

def load_embedding(word2vec_model):
    model = Word2Vec.load(word2vec_model)

    embedding = model.wv.vectors
    return embedding

def load_embedding_matrix():
    return np.load(embedding_matrix_path + '.npy')

