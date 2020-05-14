import tensorflow as tf
from layers.model_layers import Encoder, Decoder, BahdanauAttention, Pointer
from Seq2Seq.seq2seq_model import Seq2Seq
from utils.word2vec_loader import load_embedding_matrix
from PGN.decode import calc_final_dist

class PGN(tf.keras.Model):
    def __init__(self, params):
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(params['vocab_size'],
                               params['embedding_size'],
                               self.embedding_matrix,
                               params['enc_units'],
                               params['batch_size'])
        self.pointer = Pointer()
        self.attention = BahdanauAttention(params['attention_units'])
        self.decoder = Decoder(params['vocab_size'],
                               params['embedding_dim'],
                               self.embedding_matrix, 
                               params['dec_units'], 
                               params['batch_size'])
    
    def call_encoder(self, x):
        init_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_state = self.encoder(x, init_hidden)
        return enc_output, enc_state


    def call(self, enc_output, dec_hidden, enc_extended_input, target_sequence, 
             batch_oov_len, enc_padding_mask, use_coverage, prev_coverage):
        if self.params['mode'] == 'train':
            output_sequence = self._decode_target(enc_output, 
                                                  dec_hidden, 
                                                  enc_extended_input, 
                                                  target_sequence, 
                                                  batch_oov_len, 
                                                  enc_padding_mask, 
                                                  use_coverage, 
                                                  prev_coverage)
            return output_sequence

        elif self.params['mode'] == 'test':
            pass
        else:
            print('params mode must be either train or test.')
    
    def _decode_target(self, enc_output, dec_hidden, enc_extended_input, target_sequence, 
                       batch_oov_len, enc_padding_mask, use_coverage, prev_coverage):

        predictions = []
        attentions = []
        coverages = []
        p_gens = []

        # Teacher Forcing
        for t in range(target_sequence.shape[1]):
            # target_sequence[:, t] shape (batch_size, 1)
            context_vector, attention_weights, coverage = self.attention(enc_output, dec_hidden, enc_padding_mask, use_coverage, prev_coverage)
            dec_x, prediction, state = self.decoder(target_sequence[:, 0], dec_hidden, context_vector)

            p_gen = self.pointer(context_vector, state, tf.squeeze(dec_x, axis = 1))

            predictions.append(prediction)
            coverages.append(coverage)
            attentions.append(attention_weights)

            prev_coverage = coverage
        
        final_decoding_dist = calc_final_dist(enc_extended_input, predictions, attentions, p_gens, batch_oov_len, self.params['batch_size'], self.params['vocab_size'])
        return final_decoding_dist


    

