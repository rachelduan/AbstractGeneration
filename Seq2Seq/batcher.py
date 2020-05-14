from utils.data_loader import load_train_dataset,  load_test_dataset
import tensorflow as tf

def generate_train_batch(batch_size, max_enc_len = 200, max_dec_len = 50, sample_num = None):
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    if sample_num:
        train_X = train_X[:sample_num]
        train_Y = train_Y[:sample_num]
    
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder = True)

    steps_per_epoch = len(train_X) // batch_size

    return dataset, steps_per_epoch

def generate_test_batch(batch_size, max_enc_len = 200, max_dec_len = 50):
    test_X = load_test_dataset(max_enc_len)

    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    dataset = dataset.batch(batch_size, drop_remainder = False)

    return dataset

