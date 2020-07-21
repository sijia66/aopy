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
def load_ecog_clfp_data(data_file_name,t_range=(0,-1),exp_file_name=None,mask_file_name=None,compute_mask=True):

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
        data_type = np.ushort
    elif rec_type == 'lfp':
        srate = 1000
        data_type = np.float32
    elif rec_type == 'clfp':
        srate = 1000
        data_type = np.float32

    # get microdrive parameters
    microdrive_name_list = [md['name'] for md in experiment['hardware']['microdrive']]
    microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
    microdrive_dict = experiment['hardware']['microdrive'][microdrive_idx]
    num_ch = len(microdrive_dict['electrodes'])

    exp = {"srate":srate,"num_ch":num_ch}

    data_type_size = data_type().nbytes
    file_size = os.path.getsize(data_file_name)
    n_offset_samples = np.round(t_range[0]*srate)
    n_offset = n_offset_samples*data_type_size
    n_all = int(np.floor(file_size/num_ch/data_type_size))
    if t_range[1] == -1:
        n_stop = n_all
    else:
        n_stop = np.min((np.round(t_range[1]*srate),n_all))
    n_read = n_stop-n_offset_samples

    # load data
    print("Loading data file:")
    # n_offset value is the number of bytes to skip
    # n_read value is the number of items to read (by data type)
    data = read_from_file(data_file_name,data_type,num_ch,n_read,n_offset)
    if rec_type == 'raw': # correct uint16 encoding errors
        data = np.array(data,dtype=np.float32)
        for ch_idx in range(num_ch):
            is_neg = data[ch_idx,:] > 2**15
            data[ch_idx,is_neg] = data[ch_idx,is_neg] - (2**16 - 1)

    # check for mask file, load if valid, compute if not
    if os.path.exists(mask_file_name):
        with open(mask_file_name,"rb") as mask_f:
            mask = pkl.load(mask_f)
    elif compute_mask:
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
    else:
        mask = []

    return data, exp, mask

# read T seconds of data from the start of the recording:
def read_from_start(data_file_path,data_type,n_ch,n_read):
    data_file = open(data_file_path,"rb")
    data = np.fromfile(data_file,dtype=data_type,count=n_read*n_ch)
    data = np.reshape(data,(n_ch,n_read),order='F')
    data_file.close()

    return data

def read_single_trial(data_file, events, trials, 
                      ti = 0, #which trial to load 
                      FS = 1000, task_field = None, bn = np.array([-300, 500]), #timing alignment parameters
                      n_ch = 211, 
                      data_type = np.float32):
    """
    inputs:
    data_file: file object
    ti: trial number to load, corresponding to sequence in the trials structure. 

    outputs:
    data: an n_ch by n_time_points np array
    """
    
    #check if there is task field information
    if task_field is None: 
        print('task_field:str is not supplied!')
        return None

    subtrial_matlab = trials[ti]['Trial']
    #convert to 0 based indexing in python
    subtrial_py = subtrial_matlab - 1

    #where to start in bytes?
    align_at_time = events[task_field][subtrial_py]
    start_at_time = align_at_time + bn[0] 
    data_type_size = data_type().nbytes
    start_at_byte = int(round(start_at_time*FS/1e3)) * data_type_size *n_ch

    #how many samples to load?
    N1 = int(round(bn[0]*FS/1000.0))
    N2 = int(round(bn[1]*FS/1000.0))
    dN=N2-N1  #number of time points

    #retrieve the data 
    data = np.fromfile(data_file,dtype=data_type,
                        offset=start_at_byte,
                        count=dN * n_ch)
    
    #reshape the data into n_ch by dN samples
    data = np.reshape(data,(n_ch,dN),order='F')

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
