# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    with open(path_to_rttm) as f:
        lines = f.readlines()

    for line in lines:
        symbols = line.split(" ")
        start = int(float(symbols[3])*fs)
        stop = int(float(symbols[4])*fs)
        vad_markup[start:start + stop] = 1
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal
    
    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    signal_length  = signal.shape[0]
    frame_length = window
    frame_step = shift
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) # make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z) # pad Signal to make sure that all frames have equal number of samples without
                                                 # truncating any samples from the original signal

    window = np.hamming(frame_length)
    frames = np.zeros((num_frames, frame_length))
    id_start = 0
    id_end = frame_length
    cnt = 0

    while(id_end < len(pad_signal)):
        frames[cnt] = pad_signal[id_start:id_end] * window
        cnt += 1
        id_start += frame_step
        id_end += frame_step
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    E = np.zeros(frames.shape[0]).astype('float32')

    E  = np.sum(frames, axis = 1)
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = (E - np.mean(E)) / np.std(E)

    ###########################################################
    # Here is your code
    
    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    count_frames = len(E)
    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):
        ...
        # E-step
        ###########################################################
        eStep_get_g_i = lambda e: (w * gauss_pdf(e, m, sigma)) / np.sum((w * gauss_pdf(e, m, sigma)))
        g = np.array(list(map(eStep_get_g_i, E)))
        ###########################################################

        # M-step
        ###########################################################
        w = g.sum(axis=0) / count_frames
        m = (g*np.repeat([E], 3, axis=0).T).sum(axis=0) / (count_frames * w)
        sigma = np.array([(g.T[idx]*(E - m[idx])**2).sum() / (count_frames * w[idx]) for idx in range(3)])
        sigma = np.sqrt(sigma)
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    g0 = np.array([(w[0]*gauss_pdf(E[idx], m[0], sigma[0])) / (w*gauss_pdf(E[idx], m, sigma)).sum()
                   for idx in range(len(E))])
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    
    return np.convolve(signal,impulse_response, 'same')

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    signal_noise = signal + np.random.normal(0, sigma_noise, len(signal))
    
    return signal_noise