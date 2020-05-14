import numpy as np
import pandas as pd
import jieba
import re
import random
from config import *
from collections import Counter
from multicore import parallelize
from file_utils import save_dict
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import gensim

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

######################################################################################################
### Part 0. Utility functions for load processed data 
######################################################################################################
def load_train_dataset(max_enc_len, max_dec_len):
    """
    :return: load processed  training data
    """
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_label_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y


def load_test_dataset(max_enc_len=200):
    """
    :return: load processed test data
    """
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :max_enc_len]
    return test_X

######################################################################################################
### Part I. Utility functions for preprocessing data 
######################################################################################################
def load_data(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, encoding='utf-8')
    test_data = pd.read_csv(train_data_path, encoding='utf-8')

    print('train data size {},test data size {}'.format(len(train_data), len(test_data)))

    train_data.dropna(subset = ['Report'], inplace = True)

    train_data.fillna('', inplace  = True)
    test_data.fillna('', inplace = True)

    return train_data.iloc[:30,:], test_data.iloc[:30, :]


def clean_data(sentence):
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】~@#￥%……&*（）]+|车主说|技师说|语音|图片',
            '', sentence)
    else:
        return ''


def tokenize(sentence):
    stopwords = load_stopwords(stopwords_path)
    return ' '.join([word for word in jieba.lcut(sentence) if word not in stopwords])


def process_sentence(sentence):
    sentence = clean_data(sentence)

    sentence = tokenize(sentence)
    return sentence


def dataframe_process(data):
    
    for column in ['Question', 'Dialogue']:
        data[column] = data[column].apply(process_sentence)
    
    if 'Report' in data.columns:
        data['Report'] = data['Report'].apply(process_sentence)
    
    return data

def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r') as file:
        lines = file.readlines()
    
    stopwords = [line.strip() for line in lines]
    return stopwords


######################################################################################################
### Part II. Utility functions building word embeddings
######################################################################################################
def vocab_index_dict(merged_data):
    '''
    generate vocab-index pair
    merged_data: string with words separated by ' '
    '''
    word_list = merged_data.split()
    word_counts = Counter(word_list)
    vocab_list = sorted(word_counts.items(), key = lambda x: x[1], reverse = True)
    vocab_list = [(vocab_list[i][0], i+1) for i in range(len(vocab_list))]

    vocab_to_index = dict(vocab_list)

    index_word_list = [(item[1], item[0]) for item in vocab_list]
    index_to_vocab = dict(index_word_list)
    return vocab_to_index, index_to_vocab

def train_w2v_embeddings(data_path, model_path):
    model = word2vec.Word2Vec(LineSentence(data_path), workers=8, min_count=5, size=200)
    model.save(model_path)
    return model


def build_embedding_matrix(model):
    '''
    using word2vec index2word list, which is a list of string(word)
    '''
    embedding_dim = model.wv.vector_size
    vocab_size = len(model.wv.vocab)
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        embedding_matrix[i, :] = model.wv[model.wv.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    
    np.savetxt('save_embedding_matrix_path', embedding_matrix, fmt='%0.8f')
    print('embedding matrix extracted')
    return embedding_matrix

def build_embedding_matrix_direct(model):
    '''
    get embedding matrix directly from the model
    '''
    return model.wv.vectors

def get_max_len(sentences):
    '''
    sentences: Series object
    '''
    lens = sentences.str.count(' ')+1
    return lens.mean() + 2 * lens.std()

def pad_sentence(x, x_max_len, vocab):
    '''
    fill in <start> <end> <unk> <pad>
    '''
    tokens = [token if token in vocab else '<unk>' for token in x.strip().split()]
    tokens = tokens[: x_max_len]

    padded_tokens = ['<start>'] + tokens + ['<end>']
    padded_sentence = padded_tokens + ['<pad>'] * (x_max_len - len(tokens))
    return ' '.join(padded_sentence)


def transform(sentence, vocab_to_index):
    '''
    re-represent the sentence using token indices
    '''
    tokens = sentence.strip().split()
    return [vocab_to_index[token] if token in vocab_to_index.keys() else vocab_to_index['<unk>'] for token in tokens]


def build_dataset(train_data_path,  test_data_path):

    # 1. load data
    train_data, test_data = load_data(train_data_path, test_data_path)

    jieba.load_userdict(user_dict)

    # 2. parallelize
    train_data = parallelize(train_data, dataframe_process)
    test_data = parallelize(test_data, dataframe_process)
    
    # 3. save train/test data
    train_data.to_csv(processed_train_data_path, index = None, header = True)
    test_data.to_csv(processed_test_data_path, index = None, header = True)


    # 4. save merged data
    train_data['merged'] = train_data[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis = 1)
    test_data['merged'] = train_data[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis = 1)
    merged_data = pd.concat([train_data['merged'], test_data['merged']], axis = 0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_data),
                                                                               len(test_data),
                                                                               len(merged_data)))

    merged_data.to_csv(merged_data_path, index = None, header = True)

    # home-made vocab_index_dict
    # vocab_to_index, index_to_vocab = vocab_index_dict(merged_data)

    # 5. # train model and build embedding matrix
    # # NOTE: gensim library builds the vocab list in an descending order of frequency
    model = train_w2v_embeddings(merged_data_path, word2vec_model_path)
    embedding_matrix = build_embedding_matrix_direct(model)
    print('embedding matrix shape: {}'.format(embedding_matrix.shape))

    index_to_vocab = {index:word for index, word in enumerate(model.wv.index2word)}
    vocab_to_index = {word:index for index, word in enumerate(model.wv.index2word)}

    # 6. separate input data and labels
    train_data['X'] = train_data[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis = 1)
    test_data['X'] = test_data[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis = 1)

    # 7. <start> <end> <unk> padding 
    # input sentences processing
    vocab = model.wv.vocab
    train_x_max_len = get_max_len(train_data['X'])
    test_x_max_len = get_max_len(test_data['X'])
    x_max_len = max(train_x_max_len, test_x_max_len)
    train_data['X'] = train_data['X'].apply(lambda x: pad_sentence(x, x_max_len, vocab))
    test_data['X'] = test_data['X'].apply(lambda x: pad_sentence(x, x_max_len, vocab))

    # label sentences (report) processing
    train_label_max_len = get_max_len(train_data['Report'])
    train_data['Y'] = train_data['Report'].apply(lambda x: pad_sentence(x, train_label_max_len, vocab))


    # 8. save padded data for word2vec model retraining
    train_data['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_data['Y'].to_csv(train_label_pad_path, index=None, header=False)
    test_data['X'].to_csv(test_x_pad_path, index=None, header=False)

    # 9. retrain
    print('start retraining word2vec model...')
    model.build_vocab(LineSentence(train_x_pad_path), update = True)
    model.train(LineSentence(train_x_pad_path), epochs = 1, total_examples = model.corpus_count)
    print('progressing 1/3.')

    model.build_vocab(LineSentence(test_x_pad_path), update = True)
    model.train(LineSentence(test_x_pad_path), epochs = 1, total_examples = model.corpus_count)
    print('progressing 2/3.')

    model.build_vocab(LineSentence(train_label_pad_path), update = True)
    model.train(LineSentence(train_label_pad_path), epochs = 1, total_examples = model.corpus_count)
    print('finish retraining.')

    # 10. update vocab_to_index and index_to_vocab
    vocab_to_index = {token: index for index, token in enumerate(model.wv.index2word)}
    index_to_vocab = {index: token for index, token in enumerate(model.wv.index2word)}

    # 11. save results
    model.save(word2vec_model_path)
    print('final word2vec model has a vocabulary of size ', len(model.wv.vocab))

    save_dict(vocab_to_index, vocab_to_index_path)
    save_dict(index_to_vocab, index_to_vocab_path)

    embedding_matrix = model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)

    # 12. transform sentence to index sequence
    train_x_index_seq = train_data['X'].apply(lambda x: transform(x, vocab_to_index))
    test_x_index_seq = test_data['X'].spply(lambda x: transform(x, vocab_to_index))

    train_label_index_seq = train_data['Y'].apply(lambda x: transform(x, vocab_to_index))

    # save reaults
    np.save(train_x_path, train_x_index_seq)
    np.save(test_x_path, test_x_index_seq)
    np.save(train_label_path, train_label_index_seq)

    return train_x_index_seq, train_label_index_seq, test_x_index_seq




if __name__ == '__main__':
    build_dataset(train_data_path, test_data_path)