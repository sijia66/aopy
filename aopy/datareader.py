# datareader.py

# submodule containing functions for reading pesaran-style data into python

from aopy import datafilter

import numpy as np
import scipy.io as sio
from pandas import read_csv
import string
import pickle as pkl
import os
import json
import warnings


# wrapper to read and handle clfp ECOG data
def load_ecog_clfp_data(data_file_name,t_range=(0,-1),exp_file_name=None,mask_file_name=None):

    # get file path, set ancillary data file names
    data_file = os.path.basename(data_file_name)
    data_file_kern = os.path.splitext(data_file)[0]
    rec_id, microdrive_name, rec_type = data_file_kern.split('.')
    data_path = os.path.dirname(data_file_name)
    if exp_file_name is None:
        exp_file_name = os.path.join(data_path,rec_id + ".experiment.json")
    if mask_file_name is None:
        mask_file_name = os.path.join(data_path,data_file_kern + ".mask.pkl")


    # check for experiment file, load if valid, exit if not.
    if os.path.exists(exp_file_name):
#         exp = read_csv(exp_file_name)
#         srate = int(exp.srate_ECoG[0])
#         num_ch = int(exp.nch_ECoG[0])
#         exp = {"srate":srate,"num_ch":num_ch}
        with open(exp_file_name,'r') as f:
            experiment = json.load(f)
    else:
        raise NameError("Experiment file {} either invalid or not found. Aborting Process.".format(exp_file_name))

    # get srate
    if rec_type == 'raw':
        srate = experiment['hardware']['acquisition']['samplingrate']
    elif rec_type == 'lfp':
        srate = 1000
    elif rec_type == 'clfp':
        srate = 1000

    # get microdrive parameters
    microdrive_name_list = [md['name'] for md in experiment['hardware']['microdrive']]
    microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
    microdrive_dict = experiment['hardware']['microdrive'][microdrive_idx]
    num_ch = len(microdrive_dict['electrodes'])

    exp = {"srate":srate,"num_ch":num_ch}

    data_type = np.float32
    data_type_size = data_type().nbytes
    file_size = os.path.getsize(data_file_name)
    n_offset = np.round(t_range[0]*srate)*num_ch*data_type_size
    n_all = int(np.floor(file_size/num_ch/data_type_size))
    if t_range[1] == -1:
        n_stop = n_all
    else:
        n_stop = np.round(t_range[1]*srate)*num_ch*data_type_size
    n_read = n_stop-n_offset


    # load data
    print("Loading data file:")
    data = read_from_file(data_file_name,data_type,num_ch,n_offset,n_read)

    # check for mask file, load if valid, compute if not
    if os.path.exists(mask_file_name):
        with open(mask_file_name,"rb") as mask_f:
            mask = pkl.load(mask_f)
    else:
        print("No mask data file found for {0}".format(data_file))
        print("Computing data masks:")
        hf_mask,_ = datafilter.high_freq_data_detection(data,srate)
        _,sat_mask_all = datafilter.saturated_data_detection(data,srate)
        sat_mask = np.any(sat_mask_all,axis=0)
        mask = {"hf":hf_mask,"sat":sat_mask}
        # save mask data to current directory
        print("Saving mask data for {0} to {1}".format(data_file,mask_file_name))
        with open(mask_file_name,"wb") as mask_f:
            pkl.dump(mask,mask_f)

    return data, exp, mask

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
    if np.version.version >= "1.17": # "offset" field not added until later installations
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch,
                           offset=n_offset*n_ch)
    else:
        warnings.warn("'offset' feature not available in numpy <= 1.13 - reading from the top",FutureWarning)
        data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
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
