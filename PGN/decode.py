import tensorflow as tf


def calc_final_dist(enc_batch_extended_vocab, vocab_dists, attention_dists, p_gens, batch_oov_len, batch_size, vocab_size):
    '''
    enc_extended_input shape (batch_size, enput_sequence_length) 
    vocab_dist / predictions shape (decode_length, batch_size, vocab_size)
    attention_dists / attentions shape (decode_length, batch_size, input_length)
    p_gens shape (decode_length, batch_size, 1 )
    '''
    vocab_dists = [p_gen * dist for  (p_gen, dist) in zip(p_gens, vocab_dists)]
    attention_dists = [(1-p_gen) * dist for (p_gen, dist) in zip(p_gens, attention_dists)]

    extended_size = vocab_size + batch_oov_len
    extra_zeros = tf.zeros((batch_size, batch_oov_len))

    vocab_dists_extended = [tf.concat([dist, extra_zeros], axis = 1) for dist in vocab_dists]

    # batch_nums shape (batch_size, 1)
    batch_nums = tf.expand_dims(tf.range(0, batch_size), axis = 1)

    attention_len = tf.shape(enc_batch_extended_vocab)[1]
    # batch_nums shape (batch_size, attention_len)
    batch_nums = tf.tile(batch_nums, [1, attention_len])

    # indices shape (batch_size, input_sequence_length, 2)
    indices = tf.stack((batch_nums, enc_batch_extended_vocab), axis=2)
    shape = [batch_size, extended_size]

    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attention_dists]

    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists