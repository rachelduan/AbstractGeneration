import os as __os

# Project root path
__root = __os.getcwd()

# user_dict path
user_dict = __os.path.join(__root, 'data', 'user_dict.txt')

# train/test data path
train_data_path = __os.path.join(__root, 'data', 'AutoMaster_TrainSet.csv')
test_data_path  = __os.path.join(__root, 'data', 'AutoMaster_TestSet.csv')

# stop words path
stopwords_path = __os.path.join(__root, 'data', 'stopwords.txt')

# processed train and test data path
processed_train_data_path = __os.path.join(__root, 'data', 'train_data.csv')
processed_test_data_path = __os.path.join(__root, 'data', 'test_data.csv')

# processed merged data path
merged_data_path = __os.path.join(__root, 'data', 'merged_sentence.csv')

# word2vec model path
word2vec_model_path = __os.path.join(__root, 'data', 'model', 'word2vec_model')

# padded training data path
train_x_pad_path = __os.path.join(__root, 'data', 'train_x_pad.csv')

# padded training label path
train_label_pad_path = __os.path.join(__root, 'data', 'train_label_pad.csv')

# padded testing data path
test_x_pad_path = __os.path.join(__root, 'data', 'test_x_pad.csv')

# vocab to index path
vocab_to_index_path = __os.path.join(__root, 'data', 'model', 'vocab_index.txt')
index_to_vocab_path = __os.path.join(__root, 'data', 'model', 'idnex_vocab.txt')

embedding_matrix_path = __os.path.join(__root, 'data', 'model', 'embedding_matrix')

# training sequences
train_x_path = __os.path.join(__root, 'data', 'train_x')
train_label_path = __os.path.join(__root, 'data', 'train_label')
test_x_path = __os.path.join(__root, 'data', 'test_x')

# check point dir path
checkpoint_dir = __os.path.join(__root, 'data', 'checkpoints', 'training_checkpoints_mask_loss_dim500_seq')

checkpoint_prefix = __os.path.join(checkpoint_dir, 'ckpt')

# results dir
result_dir = __os.path.join(__root, 'result')

# embedding dimension
embedding_dim = 500
