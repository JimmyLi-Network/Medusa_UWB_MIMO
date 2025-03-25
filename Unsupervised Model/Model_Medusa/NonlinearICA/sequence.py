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



class StationarySequence(tf.keras.utils.Sequence):
    def __init__(self, source, input_shape, time_offset, batch_size):
        # The source data stream has the format '(numSources, sourceLength)'
        self._src = source
        self._num_src, self._src_len = source.shape
        self._time_offset = time_offset
        self._batch_size = batch_size
        self._input_shape = input_shape # Size of (numSources, windowLength)
        #self._num_epochs = num_epochs
        #self._frame_size = input_shape[1]
        #self._step_size = step_size

        # The number of samples must be equal to the batch size. We should
        # eventually be able to remove the batch_size parameter
        #assert(self._src_len == batch_size)

        # The number of input sources into the ML model must match the 
        # number of sources
        assert(input_shape[0] == 2*self._num_src)

        # Batch size needs to be even
        assert(np.mod(batch_size, 2) == 0)
        #self.seq_len = seq_len

        # This is for randomizing matches
        self._shuffle_index = None
        self.update_shuffle()

    def update_shuffle(self):
        self._shuffle_index = np.random.permutation(self._src_len)

    def input_sequence(self):
        x1 = np.roll(self._src, self._time_offset, axis=1)
        z = np.vstack((self._src, x1)).T
        return z
        
    def summary(self):
        print('**** Sequence summary ****')
        print('Input shape:({},{})\nSrc Len: {}\nNum Srcs: {}\nTime offset: {}'.format(
            self._input_shape[0], self._input_shape[1], 
            self._src_len, self._num_src, self._time_offset))
            #self.num_input_shapes(), # Num inputs
            #self.actual_batch_size(), 
            #self.num_batches())
    
    def __len__(self):
        #return int(np.ceil(self._src_len * self._src_len / self._batch_size))
        #return 1000
        #return 1

        N = int(self._batch_size/2)
        num_batches = int(np.floor(self._src_len/N))
        return 100*num_batches

    def __getitem__(self, index):

        N = int(self._batch_size/2)
        tx = np.random.permutation(self._src_len)[:N]
        ty = np.random.permutation(self._src_len)[:N]
        tz = np.random.permutation(self._batch_size)
        
        #return z, zlabel
        return self.get_sequence(N, tx, ty, tz)

    def get_sequence(self, N, tx, ty, tz):
        x0 = self._src[:, tx]
        x1 = np.roll(self._src, self._time_offset, axis=1)[:,tx]
        y1 = self._src[:, ty]

        x = np.vstack((x0, x1))
        y = np.vstack((x0, y1))

        z = np.vstack((x.T, y.T))[tz,:]
        zlabel = np.hstack((np.ones(N), np.zeros(N)))[tz]
        return z, zlabel
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def src_len(self):
        # The number of samples in each source
        return self._src_len

    @property
    def num_src(self):
        # Total number of sources in this sequence
        return self._num_src

class MultiStationarySequence(tf.keras.utils.Sequence):
    def __init__(self, seqs):
        assert(len(seqs) > 0), 'Sequence count must be non-zero'
        self._seqs = seqs
        self._num_batches = len(seqs[0])
        self._index = 0
        self._batch_size = seqs[0].batch_size
        self._src_len = seqs[0].src_len
        self._num_src = seqs[0].num_src

        # All of the sequences should have the same length
        for s in seqs:
            assert(self._num_batches == len(s)), 'All sequences must have the same number of batches'
            assert(self._batch_size == s.batch_size), 'All sequences must have the same batch size'
            assert(self._src_len == s.src_len), 'All sequences must have the same length'
            assert(self._num_src == s.num_src), 'All sequences must have the same number of sources'

    def __len__(self):
        return self._num_batches
        #return len(self._seqs) * self._num_batches
        #return self._seqlen
        #return np.min((10, len(self._seqs))) * self._seqlen

    def __getitem__(self, index):

        N = int(self._batch_size/2)
        tx = np.random.permutation(self._src_len)[:N]
        ty = np.random.permutation(self._src_len)[:N]
        tz = np.random.permutation(self._batch_size)

        def get_seq(s):
            z, zlabel = s.get_sequence(N, tx, ty, tz)
            assert(len(z.shape) == 2)
            assert(z.shape == (self._batch_size, 2*self._num_src))
            return z, zlabel

        x = [get_seq(s) for s in self._seqs] 
        z = np.stack([a for (a,b) in x], axis=1)
        assert(z.shape == (self._batch_size, len(self._seqs), 2*self._num_src))

        zlabel = x[0][1]
        return z, zlabel
        
        # r = self._seqs[self._index][int(index/len(self._seqs))] # __getitem__
        # self._index  = (self._index + 1) % len(self._seqs)
        # return r

    def input_sequence(self):
        #assert(index >= 0 and index < len(self._seqs)), 'Sequence index out of bounds'
        #return self._seqs[index].input_sequence()

        s = [s.input_sequence() for s in self._seqs]
        X = np.stack(s, axis=1)
        assert(X.shape == (self._src_len, len(self._seqs), 2*self._num_src))

        return X

    @property
    def input_shape(self):
        assert(len(self._seqs) > 0), 'Empty sequence count'
        #return self._seqs[0].input_shape
        return [len(self._seqs), 2*self._num_src]

    @property
    def sequence_count(self):
        return len(self._seqs)

    @property
    def batch_size(self):
        return self._batch_size

    def update_shuffle(self):
        for s in self._seqs:
            s.update_shuffle()

class NonStationarySequence(StationarySequence):
    def __init__(self, source, input_shape, time_offset, batch_size, num_classes):
        super.__init__(sourc=source, input_shape=input_shape, 
            time_offset=time_offset, batch_size=batch_size)

        self.num_classes = num_classes

    def __getitem__(self, index):

        # Partition the data into blocks and grab from different blocks, with labels accordingly
        pass
