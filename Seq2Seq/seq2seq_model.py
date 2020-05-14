import tensorflow as tf
import time
from layers.model_layers import Encoder, Decoder, BahdanauAttention
from utils.word2vec_loader import load_embedding_matrix
from Seq2Seq.batcher import generate_batch


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(params['vocab_size'], 
                               params['embedding_dim'],
                               self.embedding_matrix,
                               params['enc_units'],
                               params['batch_size'])
        self.attention = BahdanauAttention(params['attention_units'])
        self.decoder = Decoder(params['vocab_size'],
                               params['embedding_dim'],
                               self.embedding_matrix,
                               params['dec_units'],
                               params['batch_size'])
    
    def call_encoder(self, x):
        hidden = self.encoder.initialize_hidden_state()
        output, state = self.encoder(x, hidden)
        return output, state
    
    def call_decoder_one_step(self, x, prev_hidden, hidden_seqenuce):
        context_vector, attention_weights = self.attention(prev_hidden, hidden_seqenuce)

        prediction, state = self.decoder(x, None, context_vector)

        return prediction, state, context_vector, attention_weights
    
    def call(self, x, prev_hidden, enc_output, target):
        '''
        x shape (batch_size, 1)
        target shape (batch_size, target_length)
        '''
        predictions = []
        attentions = []

        context_vector, _ = self.attention(prev_hidden, enc_output)

        for t in range(target.shape[1]):
            prediction, state = self.decoder(x, prev_hidden, context_vector)

            context_vector, attention_weights = self.attention(state, enc_output)

            predictions.append(prediction)
            attentions.append(attention_weights)

            # teacher forcing
            # shape of target[:, t] is (batch_size,), need expansion
            x = tf.expand_dims(target[:, t], 1)
        
        # after stack shape: (batch_size, dec_sequence_length)
        return tf.stack(predictions, 1), attentions
    
