# datareader.py

# submodule containing functions for reading pesaran-style data into python

import numpy as np
import scipy.io as sio

# read T seconds of data from the start of the recording:
def read_from_start(data_file_path,data_type,n_ch,n_read):
    data_file = open(data_file_path,"rb")
    data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order='F')
    data_file.close()

    return data

# read some time from a given offset
def read_from_file(data_file_path,data_type,n_ch,n_read,n_offset):
    data_file = open(data_file_path,"rb")
    data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch,
                       offset=n_offset*n_ch)
    data = np.reshape(data,(n_ch,n_read),order='F')
    data_file.close()

    return data

# read variables from the "experiment.mat" files
def get_exp_var(exp_data,*args):
    out = exp_data.copy()
    for k, var_name in enumerate(args):
        if k > 1:
            out = out[None][0][None][0][var_name]
        
        else:
            out = out[var_name]
        
    return out

    