import tensorflow as tf
from Seq2Seq.train_helper import train_model
from Seq2Seq.seq2seq_model import Seq2Seq
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.word2vec_loader import load_vocab
from utils.config import word2vec_model_path, checkpoint_dir

def train(params):
    config_gpu()

    vocab_to_index, _ = load_vocab(word2vec_model_path)

    params['vocab_size'] = len(vocab_to_index)

    seq2seq = Seq2Seq(params)

    checkpoint = tf.train.Checkpoint(model = seq2seq)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep = 5)

    train_model(seq2seq, params, vocab_to_index, checkpoint_manager)


if __name__ == "__main__":
    params = get_params()

    train(params)