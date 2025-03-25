import scipy.io as io
import numpy as np
import pandas as pd

data_csv = pd.read_csv('data.csv')
data_numpy = np.transpose(data_csv)
#data_x_array = np.array_split(data_x, 50, axis=0)
np.save('train_x.npy',data_numpy)

breath_text = pd.read_csv('breathing.csv',usecols=['Data Set 1:Force(N)'])
breath_numpy = np.transpose(breath_text)
np.save('train_y.npy',breath_numpy)
