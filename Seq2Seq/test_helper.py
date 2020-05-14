import tensorflow as tf
from Seq2Seq.batcher import generate_test_batch

def greedy_decode(model, vocab_to_index, params):
    batch_size = params['batch_size']
    results = []

    for test_X in generate_test_batch(batch_size):
        results += greedy_decode_one_batch(model, test_X, vocab_to_index, params)
    
    return results

def greedy_decode_one_batch(model, batch_data, vocab_to_index, params):
    pass