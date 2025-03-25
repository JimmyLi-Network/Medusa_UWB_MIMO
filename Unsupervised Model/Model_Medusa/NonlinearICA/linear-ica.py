"""
Linear ICA decomposition. This is the last step after nonlinear ICA.
"""

import argparse
import pathlib
import pdb
import sys
import traceback
import datetime
import h5py
import numpy as np
import shutil

from sklearn.decomposition import FastICA

def ica(X, num_components=None):
    """
    Input parameter format:
    X: array of features with dimensions (num_features, num_samples)
    num_components: Number of ICA components. If this is not specified, it defaults to the 
                    number of features

    Return values:
    Y: individual components, in the format (num_features, num_samples)
    """

    if num_components is None:
        num_components = X.shape[1] 

    #transformer = FastICA(n_components=num_components, random_state=0, tol=1e-1, max_iter=400)
    transformer = FastICA(n_components=num_components, random_state=0)
    #Y = np.transpose(transformer.fit_transform(np.transpose(X)))
    Y = transformer.fit_transform(X)

    return Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear ICA')
    parser.add_argument('-f', '--source', required=True,
        help='Input feature data file')
    
    args = parser.parse_args()

    # Read the feature file
    print(f'Reading features from {args.source}')
    with h5py.File(args.source) as f:
        ds = np.array(f['features'])
        [num_samples, num_sources] = ds.shape
        components = ica(ds)
        # features = np.transpose(ds, (1,2,0))
        # Z = np.zeros(features.shape)
        # for i in range(features.shape[2]):
        #     b = np.squeeze(features[:,:,i])
        #     Y = ica(b)
        #     Z[:,:,i] = Y
    
    # Save the data to its own hdf5 file
    p = pathlib.Path(args.source)
    component_file = pathlib.Path(p).parent / 'feature_components.h5'
    print(f'Writing components to {component_file}')
    with h5py.File(component_file, 'w') as f:
        f.create_dataset('components', data = components)