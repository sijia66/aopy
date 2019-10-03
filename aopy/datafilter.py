# datafilter.py

# submodule containing data conditioning methods for ECoG


import numpy as np
import scipy.signal as sps
import progressbar as pb


# py version of noiseByHistogram.m - get upper and lower signal value bounds from a histogram
def histogram_defined_noise_levels( data, nbin=20 ):
    # remove data in outer bins of the histogram calculation
    hist, bin_edge = np.histogram(data,bins=nbin)
    low_edge, high_edge = bin_edge[1], bin_edge[-2]
    no_edge_mask = np.all([(data > low_edge), (data < high_edge)],axis = 0)
    data_no_edge = data[no_edge_mask]
    # compute gaussian 99% CI estimate from trimmed data
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_CI_lower, data_CI_higher = data_mean - 3*data_std, data_mean + 3*data_std
    # return min/max values from whole dataset or the edge values, whichever is lower
    noise_lower = low_edge if low_edge < data_CI_lower else min(data)
    noise_upper = high_edge if high_edge > data_CI_higher else max(data)
    
    return (noise_lower, noise_upper)


# py version of saturatedTimeDetection.m - get indeces of saturated data segments
def saturated_data_detection( data, srate, bad_channels=np.zeros(np.shape(data)[0]), adapt_tol=1e8 ,
                              win_n=20 ):
    num_ch, num_samp = np.shape(data)
    bad_all_ch_mask = np.zeros((num_ch,num_samp))
    data_rect = np.abs(data)
    mask = [bool(not x) for x in bad_channels]
    for ch_i in progressbar(np.arange(num_ch)[mask]):
        ch_data = data_rect[ch_i,:]
        θ1 = 50 # initialize threshold value
        θ0 = 0
        h, val = np.histogram(ch_data,int(np.max(ch_data)))
        val = np.floor(val)
        prob_val = h/np.shape(h)[0]
        
        # estimate midpoint between bimodal distribution for a theshold value
        while np.abs(θ1 - θ0) > adapt_tol:
            θ0 = θ1
            sub_θ_val_mask = val <= θ1
            sup_θ_val_mask = val > θ1
            sub_θ_val_mean = np.sum(np.multiply(val[sub_θ_val_mask],prob_val[sub_θ_val_mask]))/np.sum(prob_val[sub_θ_val_mask])
            sup_θ_val_mean = np.sum(np.multiply(val[not sup_θ_val_mask],prob_val[not sup_θ_val_mask]))/np.sum(prob_val[sup_θ_val_mask])
            θ1 = (sub_θ_val_mean + sup_θ_val_mean)/2
        
        # filter signal, boxcar window
        b_filt = np.ones(win_n)/win_n
        a_filt = 1
        ch_data_filt = sps.lfilter(b_filt,a_filt,ch_data)
        ch_data_filt_sup_θ_mask = ch_data_filt > θ1
        
        # get histogram-derived noise limits
        n_low, n_high = histogram_defined_noise_levels(ch_data)
        ch_data_low_mask = ch_data < n_low
        ch_data_high_mask = ch_data > n_high
        ch_data_filt_low_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_low_mask)
        ch_data_filt_high_mask = np.logical_and(ch_data_filt_sup_θ_mask,ch_data_high_mask)
        bad_all_ch_mask[ch_i,:] = np.logical_or(ch_data_filt_low_mask,ch_data_filt_high_mask)
        
        # clear out straggler values
        # I will hold off on implementing this until 
#         out_of_range_samp_mask = np.logical_or(ch_data < n_low, ch_data > n_high)
        
#         for samp_i in np.arange(samp_i)[np.logical_and(out_of_range_samp_mask,np.logical_not(bad_all_ch_mask[i,:]))]:
#             if np.abs(ch_data[samp_i]) >= θ1 and 
#             if samp_i < num_samp - srate*45:
                
#             else:

    num_bad = np.sum(bad_all_ch_mask,axis=0)
    sat_data_mask = num_bad > num_ch/2
    
    return sat_data_mask
