# datafilter.py

# submodule containing data conditioning methods for ECoG


import numpy as np
import numpy.linalg as npla
import scipy.signal as sps
import progressbar as pb


# python implementation of badChannelDetection.m - see which channels are too noisy
def bad_channel_detection( data, srate, lf_c=100, sg_win_t=8, sg_over_t=4, sg_bw = 0.5 ):
    print("Running bad channel assessment:")
    (num_ch,num_samp) = np.shape(data)
    
    # compute low-freq PSD estimate
    [fxx,txx,Sxx] = mt_sgram(data,srate,sg_win_t,sg_over_t,sg_bw)
    low_freq_mask = fxx < lf_c
    Sxx_low = Sxx[:,low_freq_mask,:]
    Sxx_low_psd = np.mean(Sxx_low,axis=2)
    
    psd_var = np.var(Sxx_low_psd,axis=1)
    norm_psd_var = psd_var/npla.norm(psd_var)
    low_var_θ = np.mean(norm_psd_var)/3
    bad_ch_mask = norm_psd_var <= low_var_θ
    
    return bad_ch_mask


# python implementation of highFreqTimeDetection.m - looks for spectral signatures of junk data
def high_freq_data_detection( data, srate, bad_channels=np.zeros(np.shape(data)[0]), lf_c=100):
    print("Running high frequency noise detection: lfc @ {0}".format(lf_c))
    [num_ch,num_samp] = np.shape(data)
    data_t = np.arange(num_samp)/srate
    
    # calculate multitaper spectrogram for each channel
    sg_win_t = 8 # (s)
    sg_over_t = sg_win_t // 2 # (s)
    sg_bw = 0.5 # (Hz)
    fxx,txx,Sxx = mt_sgram(data,srate,sg_win_t,sg_over_t,sg_bw) # Sxx: [num_ch]x[num_freq]x[num_t]
    num_freq, = np.shape(fxx)
    num_t, = np.shape(txx)
    Sxx_mean = np.mean(Sxx,axis=2) # average across all windows, i.e. numch x num_f periodogram
    
    # get low-freq, high-freq data
    low_f_mask = fxx < lf_c # Hz
    high_f_mask = np.logical_not(low_f_mask)
    low_f_mean = np.mean(Sxx_mean[:,low_f_mask],axis=1)
    low_f_std = np.std(Sxx_mean[:,low_f_mask],axis=1)
    high_f_mean = np.mean(Sxx_mean[:,high_f_mask],axis=1)
    high_f_std = np.std(Sxx_mean[:,high_f_mask],axis=1)
    
    # set thresholds for high, low freq. data
    low_θ = low_f_mean - 3*low_f_std
    high_θ = high_f_mean + 3*high_f_std
    bad_data_mask_all_ch = np.zeros((num_ch,num_samp))
    for ch_i in pb.progressbar(np.arange(num_ch)[np.logical_not(bad_channels)]):
        for t_i, t_center in enumerate(txx):
            low_f_mean_ = np.mean(Sxx[ch_i,low_f_mask,t_i])
            high_f_mean_ = np.mean(Sxx[ch_i,high_f_mask,t_i])
            if low_f_mean_ < low_θ[ch_i] and high_f_mean_ > high_θ[ch_i]:
                # get indeces for the given sgram window and set them to "bad:True"
                t_bad_mask = np.logical_and(data_t > t_center - sg_win_t/2, data_t < t_center + sg_win_t/2)
                bad_data_mask_all_ch[ch_i,t_bad_mask] = True
                
    bad_ch_θ = 0
    bad_data_mask = sum(bad_data_mask_all_ch) > bad_ch_θ
    
    return bad_data_mask


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


# multitaper spectrogram estimator
def mt_sgram(x,srate,win_t,over_t,bw):
    # x - input data
    # srate - sampling rate of x
    # win_t - length of window (s)
    # over_t - size of window overlap (s)
    # bw - frequency resolution, i.e. bandwidth
    nw = bw*win_t/2 # time-half bandwidth product
    n_taper = round(nw*2-1)
    win_n = srate*win_t
    over_n = srate*over_t
    dpss_w = sps.windows.dpss(win_n,nw,Kmax=n_taper)
    
    Sxx_m = []
    for k in range(n_taper):
        fxx,txx,Sxx_ = sps.spectrogram(x,srate,window=dpss_w[k,:],noverlap=over_n)
        Sxx_m.append(Sxx_)
        
    Sxx = np.mean(Sxx_m,axis=0)
    
    return fxx, txx, Sxx


# py version of saturatedTimeDetection.m - get indeces of saturated data segments
def saturated_data_detection( data, srate, bad_channels=np.zeros(np.shape(data)[0]), adapt_tol=1e8 ,
                              win_n=20 ):
    print("Running saturated data segment detection:")
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

