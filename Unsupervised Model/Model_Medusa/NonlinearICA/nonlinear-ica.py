import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import tensorflow_addons as tfa
tf.config.run_functions_eagerly(True)
assert tf.executing_eagerly()

import argparse
import pathlib
import pdb
import sys
import traceback
import datetime
import h5py
import numpy as np
import shutil
import gentest
from sequence import StationarySequence, MultiStationarySequence

# class MaxIndex(tf.keras.layers.Layer):
#     def __init__(self):
#         super(MaxIndex, self).__init__()
# 
#     def call(self, x, y):
# 

class ICAModel(tf.keras.Model):
    def __init__(self, input_shape = None, feature_size = None, num_classes = 2, training=True):
        super(ICAModel, self).__init__()
        self.encoder = None
        self.classifier = None
        self.feature_size = feature_size
        self.epoch_size = None
        self.train_index = 1 

        # The input shape should be equal to the number of channels
        #assert(len(input_shape) == 1)

        assert(num_classes >= 2)
        self.num_classes = num_classes

        self.encoder = self.create_encoder(input_shape, training)
        self.encoder.summary()
        self.classifier = self.create_classifier(self.encoder.get_layer('output').output_shape[1:])

        # if saved_model is not None:
        #     # Load the saved model from the files
        #     self.encoder.load_weights(saved_model['encoder'])
        #     self.classifier.load_weights(saved_model['classifier'])
            
        #self.encoder.summary()
        self.classifier.summary()

    def load_weights(self, encoder_weights_file, classifier_weights_file):
        self.encoder.load_weights(encoder_weights_file)
        self.classifier.load_weights(classifier_weights_file)

    def evaluate(self, x):
        features = self.encoder(x)
        prediction, weights = self.classifier(features)

        return features, prediction, weights

    @property
    def encoder_size(self):
        return self.feature_size

    def create_encoder(self, input_shape, training=True):
        def encoder_layer(input, output_layer=False):
            maxout_dim = 2
            #encoder_shape = int(input_shape[0]/2)
            encoder_shape = input_shape[-1]
            maxout_axis = len(input_shape)
            #encoder_shape = 40
            x = input
            #x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim = encoder_shape)(x, x)
            x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim = encoder_shape)(x, x)
            x = tf.keras.layers.Dense(encoder_shape*maxout_dim)(x)
            #x = tfa.layers.Maxout(encoder_shape, axis=maxout_axis)(x)

            #x = tfa.layers.Maxout(input_shape[0], axis=1)(x)
            if output_layer:
                #x = tf.keras.layers.BatchNormalization(axis=2, name='output')(x, training=training)
                x = tfa.layers.Maxout(encoder_shape, axis=maxout_axis, name='output')(x)
                #x = tfa.layers.Maxout(1, axis=1, name='output')(x)
                #x = tf.keras.layers.Attention(name='output')([x,x,x])
                #x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim = encoder_shape, name='output')(x, x)

            else:
                #x = tf.keras.layers.BatchNormalization(axis=2)(x, training=training)
                x = tfa.layers.Maxout(encoder_shape, axis=maxout_axis)(x)
                #x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim = encoder_shape)(x, x)
                #x = tf.keras.layers.Attention()([x,x,x])

                
            # else:
            #     x = tfa.layers.Maxout(encoder_shape, axis=maxout_axis)(x)

            return x

        input = tf.keras.Input(shape=input_shape, name='input')
        x = encoder_layer(input)
        x = encoder_layer(x)
        x = encoder_layer(x)
        x = encoder_layer(x)
        x = encoder_layer(x)
        x = encoder_layer(x, output_layer=True)

        #x, weights = tf.keras.layers.Attention(name='output')([x,x,x], return_attention_scores=True)
        #x = tfa.layers.Maxout(1, axis=1)(x)
        #x = tf.keras.layers.Reshape(target_shape=(input_shape[1],), name='output')(x)

        #output = [x, weights]
        output = x

        #x = encoder_layer(x)
        # output = encoder_layer(x, output_layer=True)

        encoder = tf.keras.Model(inputs=input, outputs=output, name='encoder')
        return encoder

    def create_classifier(self, input_shape):
        alpha = 0.3
        classifier_units = self.encoder_size
        
        input = tf.keras.Input(shape=input_shape, name='input')
        x = input
        x = tf.keras.layers.Dense(input_shape[-1])(x)
        x = tf.keras.layers.Dense(input_shape[-1])(x)
        
        if self.num_classes == 2:
            # x = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
            labels = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            x = labels
            x = tfa.layers.Maxout(1, axis=1)(x)
            x = tf.keras.layers.Reshape(target_shape=(1,), name='output')(x)

        else:
            output = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='output')(x)

        output = [x, labels]
        classifier = tf.keras.Model(inputs=input, outputs=output, name='classifier')
        return classifier

    def checkpoint(self, checkpoint_path, monitor_var, epoch_size, epoch_start = 1):
        self.epoch_size = epoch_size
        self.monitor_var = monitor_var
        self.checkpoint_path = checkpoint_path
        self.train_index = 1
        self.recent_monitor_value = np.finfo('f').max
        self.epoch_index = epoch_start

    # Training step
    def train_step(self, data):
        z, zlabel = data

        with tf.GradientTape() as tape:
            features = self.encoder(z)
            c, clabels = self.classifier(features)
            loss = self.compiled_loss(zlabel, c, regularization_losses=self.losses)
        learnable_params = (self.encoder.trainable_variables + self.classifier.trainable_variables)
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))
        self.compiled_metrics.update_state(zlabel, c)
        metrics = {m.name: m.result() for m in self.metrics}

        # Save the model every epoch
        if (self.epoch_size is not None) and (self.train_index % self.epoch_size == 0):
            # If the latest value of the loss variable is lower, save this model
            if self.monitor_var in metrics:
                v = metrics[self.monitor_var].numpy()
                if v < self.recent_monitor_value:

                    print(f'\n{self.monitor_var} is reduced from {self.recent_monitor_value:.4f} to {v:.4f}')
                    self.recent_monitor_value = v
                    encoder_file = f'encoder-epoch_{self.epoch_index:03d}-{self.monitor_var}_{v:.4f}.h5'
                    encoder_ckpt = self.checkpoint_path / encoder_file
                    classifier_file = f'classifier-epoch_{self.epoch_index:03d}-{self.monitor_var}_{v:.4f}.h5'
                    classifier_ckpt = self.checkpoint_path / classifier_file
                    print(f'Saving model to {encoder_file} and {classifier_file}')
                    self.encoder.save_weights(str(encoder_ckpt))
                    self.classifier.save_weights(str(classifier_ckpt))

            feature_file = f'feature-epoch_{self.epoch_index:03d}.h5'
            feature_ckpt = self.checkpoint_path / feature_file
            print(f'\nSaving encoder features to {str(feature_ckpt)}')
            with h5py.File(str(feature_ckpt), 'w') as f:
                f.create_dataset('features', data=features.numpy())
                f.create_dataset('feature_labels', data=clabels.numpy())
                # f.create_dataset('feature_weights', data=feature_weights.numpy())
                f.create_dataset('label', data=zlabel)
            self.epoch_index += 1
        self.train_index += 1
        return metrics
        
    def __str__(self):
        self.encoder.summary()
        self.classifier.summary()

    def summary(self):
        self.encoder.summary()
        self.classifier.summary()


######################################################################################
# Train the model using the sample input sequence
######################################################################################
def train_stationary_model(seq, epochs, feature_size, checkpoint_dir = None, 
        epoch_start = 1, encoder_weights=None, classifier_weights=None):
    icamodel = ICAModel(input_shape = seq.input_shape, feature_size = feature_size)
    if encoder_weights is not None and classifier_weights is not None:
        icamodel.load_weights(encoder_weights, classifier_weights)
    
    #ica = ICA(input_shape = seq.input_shape, feature_size = seq.num_samples)
    #ica = ICA(num_streams = 5, input_size = seq.tensor_length)
    #ica.train_classifier(seq, val_seq, epochs)

    # Setup the training rate and optimizer (SGD) for use during training.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=5e5,
        decay_rate=0.999
    )
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam()

    icamodel.compile(optimizer=optimizer, 
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[[tf.keras.metrics.BinaryCrossentropy(from_logits=False), 'accuracy']])

    # Tensorboard callback so that metrics can be easily viewed on tensorboard.
    base_dir = pathlib.Path.cwd()
    tensorboard_dir = base_dir / 'logs' / 'fit' 
    if not tensorboard_dir.exists():
        tensorboard_dir.mkdir(parents=True)
    log_dir = str(tensorboard_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tensorboard_callback.set_model(icamodel)

    if checkpoint_dir is None:
        ckpt_dir = base_dir / 'checkpoint'
    else:
        ckpt_dir = checkpoint_dir 

    # # assert(not ckpt_dir.exists()), f'{ckpt_dir} exists. Delete?'
    # if clear_checkpoint and ckpt_dir.exists():
    #     shutil.rmtree(ckpt_dir)

    assert(not ckpt_dir.exists()), f'Checkpoint folder exists: {ckpt_dir}'
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)

    update_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end = lambda epoch, logs: seq.update_shuffle())

    icamodel.checkpoint(ckpt_dir, 'loss', len(seq), epoch_start)
    icamodel.fit(seq, epochs=epochs, verbose='auto', 
        callbacks=[tensorboard_callback, update_callback],
        use_multiprocessing=False)

    # Write a JSON file at the end of the training
    
    print('*** Training completed ***')
    
    return icamodel

# # Trains on multiple sequences
# def parallel_training(seq, epochs, tags, feature_size):
# 
#     base_dir = pathlib.Path.cwd()
# 
#     for t in tags:
#         ckpt_dir = base_dir / pathlib.Path('checkpoint') / pathlib.Path(t)
#         encoder_weights = None
#         classifier_weights = None
#         if ckpt_dir.exists():
#             pass
# 
        # Load the most recent checkpoint from the dir


def extract_encoder_features(seq, feature_size, encoder_weights_file, classifier_weights_file):
    icamodel = ICAModel(input_shape = seq.input_shape, 
        feature_size = feature_size, training=False)
    icamodel.load_weights(encoder_weights_file, classifier_weights_file)

    features, prediction, weights = icamodel.evaluate(seq.input_sequence())
    
    return icamodel, features, prediction, weights

# def extract_encoder_multi_features(seq, feature_size, encoder_weights_file, classifier_weights_file):
#     icamodel = ICAModel(input_shape = seq.input_shape, 
#         feature_size = feature_size)
#     icamodel.load_weights(encoder_weights_file, classifier_weights_file)
# 
#     prediction = []
#     features = []
#     for s in range(seq.sequence_count):
#         f, p = icamodel.evaluate(seq.input_sequence(s))
#         prediction.append(p)
#         features.append(f)
#     
#     return icamodel, prediction, features



######################################################################################
# Helper functions to load radar data
######################################################################################
def read_complex_hdf5(f, dataset):
    #f = h5py.File(fn)
    ds_real = dataset + '/I'
    ds_imag = dataset + '/Q'
    print('Reading {}'.format(ds_real))
    data_real = np.zeros(f[ds_real].shape)
    print('Reading {}'.format(ds_imag))
    data_imag = np.zeros(f[ds_imag].shape)
    f[ds_real].read_direct(data_real)
    f[ds_imag].read_direct(data_imag)
    d = data_real + 1j*data_imag
    return d

def read_target_channels(fn):
    f = h5py.File(fn)

    # Read raw complex target track data. 
    # The data has a format (numTx, numRx, numFrames, numChannels)
    data = read_complex_hdf5(f, '/data')

    return data
    
    
######################################################################################
# Main interface into the ML model
######################################################################################
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate bio signals from movement')
    parser.add_argument('-f', '--source', 
        help='Input radar data file')
    parser.add_argument('-c', '--classifier', 
        help='Saved classifier model name')
    parser.add_argument('-e', '--encoder', 
        help='Saved encoder model name')
    parser.add_argument('-n', '--epochs', type=int, default=100,
        help='Number of epochs used to train the model')
    parser.add_argument('-t', '--test', action='store_true',
        help='Run model with a test case')
    parser.add_argument('-d', '--delay', type=int, default=1,
        help='Time offset in contrastive set')
    # parser.add_argument('--nseq', type=int,
    #     help='Number of sequences to use for training')
    # parser.add_argument('--seqoffset', type=int, default=0,
    #     help='ID of the sequence to start')
    parser.add_argument('--txrange', type=int, nargs=2,
        help='Index range of transmitters to use for training, in the format [start, end]')
    parser.add_argument('--rxrange', type=int, nargs=2,
        help='Index range of receivers to use for training, in the format [start, end]')
    parser.add_argument('--name', 
        help='Unique label given to this training run.')
    parser.add_argument('--resume', action='store_true',
        help='Continue training from a previous checkpoint')


    args = parser.parse_args()
    assert((not (args.encoder or args.classifier)) or (args.encoder and args.classifier)), \
        'Must specify both encoder and classifier model checkpoint'
    # if args.nseq:
    #     assert(args.nseq > 0), 'ERROR: Number of sequences must be positive'
    # assert(args.seqoffset >= 0), 'ERROR: Sequence offset must be non-negative'
    assert((not args.resume) or (args.encoder and args.classifier)), 'Missing checkpoint to continue'

    # Source parameters
    batch_size = 512
    feature_size = batch_size
    time_offset = args.delay
    samples_per_second = 40 # samples per second
    data_sample_offset = samples_per_second * 40
    tags = None
    if args.test:

        # If we are running a test evaluation, then the data file
        # previously used to train the model must be loaded. 
        # Otherwise, we can generate new random data.
        if args.encoder and args.classifier:
            simfile = pathlib.Path('ica-test-stationary.h5')
            if not simfile.exists():
                print('ERROR: Simulated data file not found')
                sys.exit(-1)

            print(f'Loading simulated data from {simfile}')

            with h5py.File(str(simfile)) as f: 
                data_norm = np.zeros(f['output_norm'].shape)
                f['output_norm'].read_direct(data_norm)
            num_srcs = data_norm.shape[0]
            input_shape = (2*num_srcs,)
        else:
            num_srcs = 3
            source, data, testinfo = gentest.generate_stationary(num_srcs, 4*batch_size)
            data_norm = data / np.max(data)
            with h5py.File('ica-test-stationary.h5', 'w') as f:
                f.create_dataset("output", data=data)
                f.create_dataset("source", data=source)
                f.create_dataset("output_norm", data=data_norm)

            #assert(np.all(data.shape == (5000, 5)))
            input_shape = (2*num_srcs, ) # (2*num_srcs, 1) 
            #feature_size = data_norm.shape[1] # About 20 seconds of data

        seq = StationarySequence(data_norm, input_shape, time_offset, batch_size)
    else:
        if not args.source:
            print('Missing source data file') 
            sys.exit(-1)
        data = np.abs(read_target_channels(args.source))
        
        # Normalize the data. We normalize each tx-rx pair separately
        for tx_idx in range(data.shape[0]):
            for rx_idx in range(data.shape[1]):
                d = np.squeeze(data[tx_idx, rx_idx,:,:])
                data[tx_idx,rx_idx,:,:] /= np.max(d)
        #data /= np.max(data)
        def node_iter(tx_range = None, rx_range = None):
            txr = range(data.shape[0])
            rxr = range(data.shape[1])
            if tx_range is not None:
                assert(tx_range[0] < tx_range[1]), f'Invalid Tx range {tx_range}'
                txr = range(tx_range[0], tx_range[1])

            if rx_range is not None:
                assert(rx_range[0] < rx_range[1]), f'Invalid Rx range {rx_range}'
                rxr = range(rx_range[0], rx_range[1])

            # for tx_idx in range(data.shape[0]):
            #     for rx_idx in range(data.shape[1]):
            #         yield tx_idx, rx_idx

            for tx_idx in txr:
                for rx_idx in rxr:
                    yield tx_idx, rx_idx

        def construct_sequences(tx_range = None, rx_range = None):
            for [tx_idx, rx_idx] in node_iter(tx_range, rx_range):
            # for tx_idx in range(data.shape[0]):
            #     for rx_idx in range(data.shape[1]):
                    #d = np.transpose(np.squeeze(data[tx_idx, rx_idx,:,:]))

                    #offset = 2250
                d = np.transpose(np.squeeze(data[tx_idx, rx_idx,:,:]))

                # StationarySequence.__init__(self, source, input_shape, time_offset, batch_size):
                #input_shape = (2*data.shape[3], feature_size)
                #input_shape = (2*data.shape[3], feature_size)
                input_shape = (2*data.shape[3],)

                seq = StationarySequence(d, 
                    input_shape = input_shape,
                    time_offset=time_offset,
                    batch_size=batch_size)

                yield seq 

        sequences = [s for s in construct_sequences(args.txrange, args.rxrange)]
        #tags = [f'tx_{t}-rx_{r}' for [t,r] in node_iter()]
        tags = np.array([[i,j] for i,j in node_iter(args.txrange, args.rxrange)])
        print(f'Using sequences (tx,rx): {tags}')
        seq = MultiStationarySequence(sequences)
        # if args.nseq:
        #     end_idx = args.seqoffset + args.nseq
        #     print(f'Using sequence [{args.seqoffset}:{end_idx}]')
        #     seq = MultiStationarySequence(sequences[args.seqoffset:(args.seqoffset + args.nseq)])
        # else:
        #     seq = MultiStationarySequence(sequences[args.seqoffset:])

    model = None
    if args.encoder and args.classifier:
        # We are loading a previously trained model
        base_dir = pathlib.Path.cwd()
        features_file = base_dir / 'encoder_features.h5'
        
        print('Retrieving encoder features...')
        model, features, prediction, weights = extract_encoder_features(seq, 
            feature_size, args.encoder, args.classifier)

        with h5py.File(features_file, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('prediction', data=prediction)
            f.create_dataset('weights', data=weights)
            f.create_dataset('indices', data=tags)

        print(f'Features saved to {features_file}')

    else:
        # Create the checkpoint directory
        assert(args.name), 'Model name is missing'
        checkpoint_dir = pathlib.Path.cwd() / f'checkpoint-{args.name}'
        model = train_stationary_model(seq, args.epochs, feature_size, checkpoint_dir=checkpoint_dir)
