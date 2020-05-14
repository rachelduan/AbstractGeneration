import argparse

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_enc_len', default=200, help='Encoder input max sequence length', type=int)
    parser.add_argument('--max_dec_len', default=41, help='Decoder input max sequence length', type=int)
    parser.add_argument('--batch_size', default=32, help='batch size', type=int)
    parser.add_argument('--epochs', default=1, help='train epochs', type=int)

    parser.add_argument('--beam_size', default=3,
                        help='beam size for beam search decoding (must be equal to batch size in decode mode)',
                        type=int)
    parser.add_argument('--embedding_dim', default=500, help='Words embeddings dimension', type=int)
    parser.add_argument('--enc_units', default=512, help='Encoder GRU cell units number', type=int)
    parser.add_argument('--dec_units', default=512, help='Decoder GRU cell units number', type=int)
    parser.add_argument('--attention_units', default=20, help='[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights',
                        type=int)

    parser.add_argument('--learning_rate', default=1e-3, help='Learning rate', type=float)
    parser.add_argument('--checkpoints_save_steps', default=5, help='Save checkpoints every N steps', type=int)

    args = parser.parse_args()
    ##
    # vars() method takes only one parameter and that too is optional. 
    # It takes an object as a parameter which may be can a module, a class, an instance, or any object having __dict__ attribute.
    # The method returns the __dict__ attribute for a module, class, instance, or any other object if the same has a __dict__ attribute. 
    # If the object fails to match the attribute, it raises a TypeError exception. 
    ##
    params = vars(args)

    return params


if __name__ == "__main__":
    print(get_params())
