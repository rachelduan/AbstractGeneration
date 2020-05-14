import tensorflow as tf
from utils.gpu_utils import config_gpu
from utils.config import word2vec_model_path
from utils.word2vec_loader import load_vocab, load_embedding

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_unit = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim, 
                                                   weights = [embedding_matrix], 
                                                   trainable = False)
        self.gru = tf.keras.layers.GRU(self.enc_units, 
                                       return_sequence = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
    
    def call(self, x, hidden = None):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    '''
    et = V.T * tanh(Wh.T * enc_hidden + Ws * dec_hidden)
    attetion weights = softmax(et)
    '''
    def __init__(self, units):
        self.Wh = tf.keras.layers.Dense(units)
        self.Ws = tf.keras.layers.Dense(units)
        self.Wi = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, enc_pad_mask, use_coverage, prev_coverage):
        # query: the last hidden state in the decoder, shape: (batch_size, dec_units)
        # values: sequences of hidden states of the encoder, shape: (batch_size, senquence_length, enc_units)

        # query_with_time_axis shape (batch_size, 1, dec_units)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        '''
        If your input has shape (batch_size, sequence_length, dim), 
        then the dense layer will first flatten your data to shape (batch_size * sequence_length, dim) 
        and then apply a dense layer as usual. 
        The output will have shape (batch_size, sequence_length, hidden_units). 
        '''
        if use_coverage and prev_coverage is not None:
            score = self.V(tf.nn.tanh(self.Wh(values)) + self.Ws(query_with_time_axis) + self.Wi(prev_coverage))
        else:
            # score shape (batch_size, sequence_length, 1)
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        
        mask = tf.cast(enc_pad_mask, dtype = score.dtype)
        # masked_score shape (batch_size, max_sequence_length)
        masked_score = tf.squeeze(score, axis = -1) * mask
        # masked_score shape (batch_size, max_sequence_length, 1)
        masked_score = tf.expand_dims(masked_score, axis = 2)

        attention_weights = tf.nn.softmax(score, axis = 1)

        if use_coverage and prev_coverage is not None:
            coverage = prev_coverage + attention_weights
        else if use_coverage:
            coverage = attention_weights

        # before reduce_sum shape (batch_size, senquence_length, enc_units)
        context_vextor = attention_weights * values
        # context_vector shape (batch_size, enc_units)
        context_vextor = tf.reduce_sum(context_vextor, axis = 1)

        return context_vextor, attention_weights, coverage


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        self.batch_size = batch_size
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim, 
                                                   weights = [embedding_matrix], 
                                                   trainable = False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequence = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')

        self.fullconnect = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, hidden, context_vector):
        # x shape (batch_size, 1) => (batch_size, 1, embedding_dim)
        # context vector shape (batch_size, enc_units)
        x = self.embedding(x)

        # x shape (batch_size, 1, embedding_dim+enc_units)
        dec_input = tf.concat([x, tf.expand_dims(context_vector, 1)], axis = -1)
        output, state = self.gru(dec_input, initial_state = hidden)
        # (batch_size, 1, dec_units) => (batch_size * 1, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # prediction shape (batch_size, 1)
        prediction = self.fullconnect(output)

        return dec_input, prediction, state

class Pointer(tf.keras.layers.Layer):
    '''
    Pointer layer: generate Pgen for each time step.
    Pgen = sigm(Wh.T * context_vector + Ws.T * dec_hidden + Wx.T * dec_input)
    '''
    def __init__(self):
        self.w_s = tf.keras.layers.Dense(1)
        self.w_i = tf.keras.layers.Dense(1)
        self.w_c = tf.keras.layers.Dense(1)
    
    def call(self, context_vector, dec_hidden, dec_inputs):
        return tf.nn.sigmoid(self.w_s(dec_hidden) + self.w_c(context_vector) + self.w_i(dec_inputs))


if __name__ == '__main__':
    config_gpu()

    vocab_to_index, index_to_vocab = load_vocab(word2vec_model_path)

    embedding = load_embedding(word2vec_model_path)
    vocab_size = len(vocab_to_index)

    # config model hyper-parameters
    UNITS = 1024
    SEQUENCE_LEN = 250
    BATCH_SIZE = 64
    EMBEDDING_DIM = 500

    # test encoder
    encoder = Encoder(vocab_size, EMBEDDING_DIM, embedding, UNITS, BATCH_SIZE)
    sample_input = tf.ones((BATCH_SIZE, SEQUENCE_LEN))
    sample_hidden = encoder.initialize_hidden_state()

    output, state = encoder(sample_input, sample_hidden)
    print('encoder output shape: {}, encoder state shape: {}.'.format(output.shape, state.shape))

    # test attention
    attention = BahdanauAttention(10)

    context_vector, attention_weights = attention(state, output)
    print('attention output shape: {}, encoder state shape: {}.'.format(context_vector.shape, attention_weights.shape))

    # test decoder
    sample_dec_seq = tf.random.uniform((BATCH_SIZE, 1), dtype = tf.int32)
    decoder = Decoder(vocab_size, EMBEDDING_DIM, embedding, UNITS, BATCH_SIZE)
    prediction, dec_state = decoder(sample_dec_seq, state, context_vector)



    