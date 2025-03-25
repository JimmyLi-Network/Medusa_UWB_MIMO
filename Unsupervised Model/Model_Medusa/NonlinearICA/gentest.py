
import os
from re import A
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
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
import neurokit2 as nk
#from statsmodels.tsa.arima_process import ArmaProcess


def generate_stationary(num_sources, num_samples):

    sampling_rate = 40 # Hz
    duration = int(np.ceil(num_samples / sampling_rate)) # in seconds
    output_mul = 1 
    #num_samples = sampling_rate * duration * output_mul
    source = np.zeros([num_sources, num_samples])

    def AR1(num_samples, p=0.7):
        scale = 5.0
        r = np.random.normal(scale=scale, size=num_samples)
        s = np.zeros(num_samples)
        s[0] = np.random.uniform(0, scale, 1)
        for i in range(1, num_samples):
            s[i] = (p*s[i-1]) + r[i]
        return s

    #layer_size = 2*num_sources 
    #input_shape = (None, num_sources, num_samples)
    input_shape = (num_sources,)
    input = tf.keras.Input(shape=input_shape, name='input')
    
    encoder_units = num_sources 
    alpha = 0.3
    x = tf.keras.layers.Dense(encoder_units)(input)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dense(encoder_units)(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dense(encoder_units)(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dense(encoder_units)(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dense(encoder_units)(x)
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.Dense(encoder_units)(x)
    output = tf.keras.layers.LeakyReLU(alpha, name='output')(x)

    model = tf.keras.Model(inputs=input, outputs=output, name='encoder')

    # s = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate, respiratory_rate=22)
    # s /= np.max(np.abs(s))
    # #s = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=50+(10*i))
    #     #s = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate, respiratory_rate=13+(17*i))
    # sp = s+1.1*np.abs(np.min(s))
    # source[0,:] = sp[:num_samples] #G(ArmaProcess(ar, ma).generate_sample(nsample = num_samples))

    
    #G = lambda u: -np.abs(u)
    for i in range(0, num_sources):
        # s = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=75+(10*i))
        s = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate, respiratory_rate=11 + (7*i))
        s /= np.max(np.abs(s))

        #s = nk.rsp_simulate(duration=duration, sampling_rate=sampling_rate, respiratory_rate=75+(17*i))
        sp = s+1.1*np.abs(np.min(s))
        source[i,:] = sp[:num_samples] #G(ArmaProcess(ar, ma).generate_sample(nsample = num_samples))
        # The sources must have positive values, as the feature encoder has a leaky ReLU activation function.
        # It is thus unable to represent negative valued-sources.

        #source[i,:] = np.abs(AR1(num_samples))

    data = np.array(model(source.T)).T
    #data = source

    info = {'duration':duration, 'sampling_rate':sampling_rate}

    # Normalize
    #data = data / data.max()

    # return matrix is of size (num_samples, num_sources)
    return source, data, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nonlinear test cases')
    parser.add_argument('-n', '--sources', type=int, default=1,
        help='Number of sources')
    
    args = parser.parse_args()
    [source, data, info] = generate_stationary(args.sources)
    with h5py.File('testdata.h5', 'w') as f:
        f.create_dataset('source', data=source)
        f.create_dataset('data', data=data)

    #from matplotlib import pyplot as plt