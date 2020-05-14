import tensorflow as tf
import time
from Seq2Seq.batcher import generate_train_batch

def train_model(model, params, vocab_to_index, checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']

    pad_index = vocab_to_index['<pad>']
    unk_index = vocab_to_index['<unk>']
    start_index = vocab_to_index['<start>']

    params['vocab_size'] = len(vocab_to_index)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

    def compute_loss(real, pred):
        # positions where <pad>s and <unk>s appear in real 
        # will not count in loss  
        pad_mask = tf.math.equal(real, pad_index)
        unk_mask = tf.math.equal(real, unk_index)

        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, unk_mask))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype = loss_.dtype)

        loss = mask * loss_

        return tf.reduce_mean(loss)
    
    @tf.function
    def train_one_batch(enc_input, dec_target):
        '''
        do encode + attention + decode the whole sequence
        '''
        batch_loss = 0
        with tf.GradientTape() as t:
            output, last_encode_hidden = model.call_encoder(enc_input)

            dec_first_input = tf.expand_dims([start_index] * batch_size, 1)

            predictions, _ = model(dec_first_input, last_encode_hidden, output, dec_target)

            # first element of dec_target is <start>, which is skipped by the decoder
            batch_loss = compute_loss(dec_target[:, 1:], predictions)

            variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
            gradients = t.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
    
    dataset, steps_per_epoch = generate_train_batch(batch_size)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch_index, (inputs, targets)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_one_batch(inputs, targets)
            total_loss += batch_loss

            if batch_index % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch_index,
                                                             batch_loss.numpy()))
            
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

