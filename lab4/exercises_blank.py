# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from tqdm.auto import tqdm
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
import os
from tqdm.contrib import tenumerate

def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []


    for score, label in zip(all_scores, all_labels):
        if label:
            tar_scores.append(score)
        else:
            imp_scores.append(score)

    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    argsort = all_scores.argsort()
    all_scores_sort = all_scores[argsort]
    ground_truth_sort = all_labels[argsort] == 1
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf / imp_gauss_pdf)
 
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fpr_thr = np.zeros(len_thr)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]  # decision

        err = (solution != ground_truth_sort)  # error vector

        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(
            imp_scores)  # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)

    idx = fpr_thr.argmin()
    fpr = fpr_thr[idx]
    thr = LLR[idx]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11, return_P_err=False, LLR_steps=1000):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code

    # thr = np.log((C01 - C11) * (1 - P_Htar) / ((C10 - C00) * P_Htar))
    #
    # idx = np.where(LLR > thr)[0][0]
    # solution = LLR > LLR[idx]  # decision
    # err = (solution != ground_truth_sort)  # error vector
    #
    # tpr = np.sum(err[~ground_truth_sort]) / len(tar_scores)
    # tnr = np.sum(err[ground_truth_sort]) / len(imp_scores)
    # fnr = np.sum(err[ground_truth_sort]) / len(tar_scores)
    # fpr = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    #
    # AC = C00 * tpr * P_Htar + \
    #      C10 * fnr * P_Htar + \
    #      C01 * fpr * (1 - P_Htar) + \
    #      C11 * tnr * (1 - P_Htar)

    len_thr = LLR_steps
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)
    P_err = np.zeros(len_thr)
    # LLR_steps = np.linspace(LLR.min(), LLR.max(), LLR_steps)
    LLR_steps = np.linspace(0.7, 1.5, LLR_steps)

    bayes = lambda tpr, fpr, fnr, tnr: C00 * tpr * P_Htar \
                                       + C10 * fnr * P_Htar \
                                       + C01 * fpr * (1 - P_Htar) \
                                       + C11 * tnr * (1 - P_Htar)

    for idx in range(len_thr):
        solution = LLR > LLR_steps[idx]  # decision

        err = (solution != ground_truth_sort)  # error vector

        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(
            tar_scores)  # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        tpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(
            tar_scores)  # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(
            imp_scores)  # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        tnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(
            imp_scores)  # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)

        P_err[idx] = bayes(tpr_thr[idx],
                           fpr_thr[idx],
                           fnr_thr[idx],
                           tnr_thr[idx])

    P_err_idx = np.argmin(P_err)  # argmin of error's prob.

    P_err_min = P_err.min()

    if return_P_err:
        return LLR_steps[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min, P_err
    return LLR_steps[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min
    ###########################################################
    
    # return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, LLR_steps, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    import pandas as pd
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    plot_2d = np.zeros((len(P_Htar_thr), len(P_Htar_thr)))
    res = []
    for idx, P_Htar in tenumerate(P_Htar_thr):
        llr, fnr, fpr, p_err_min, p_err = bayes_test(ground_truth_sort,
                                                     LLR, len(P_Htar_thr), tar_scores, 
                                                     imp_scores, P_Htar, 
                                                     C00, C10, C01, C11, 
                                                     is_plot=False, return_P_err=True)
        plot_2d[idx] = p_err
        res.append({
            "LLR": llr,
            "fnr": fnr,
            "fpr": fpr,
            "p_err_min": p_err_min,
            "h_tar": P_Htar
        })
    res = pd.DataFrame(res)
    res = res.sort_values("p_err_min", ascending=False)
    imshow(plot_2d)
    thr = res.head(1)["LLR"].values[0]
    fnr = res.head(1)["fnr"].values[0]
    fpr = res.head(1)["fpr"].values[0]
    AC = res.head(1)["p_err_min"].values[0]
    P_Htar = res.head(1)["h_tar"].values[0]
    
    return thr, fnr, fpr, AC, P_Htar
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import Dataset

from common import loadWAV, AugmentWAV
from ResNetBlocks import *
from preproc import PreEmphasis
    
class ResNet(nn.Module):
    # ResNet model for speaker recognition

    def __init__(self, block, layers, activation, num_filters, nOut, encoder_type='SP', n_mels=64, log_input=True, **kwargs):
        
        super(ResNet, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.torchfb        = torch.nn.Sequential(PreEmphasis(), 
                                                  torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                                       n_fft=512, 
                                                                                       win_length=400, 
                                                                                       hop_length=160, 
                                                                                       window_fn=torch.hamming_window, 
                                                                                       n_mels=n_mels))
        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1  = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(num_filters[0])
        self.relu   = activation(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2, activation=activation)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(nn.Conv1d(num_filters[3]*outmap_size, 128, kernel_size=1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(128), 
                                       nn.Conv1d(128, num_filters[3]*outmap_size, kernel_size=1), 
                                       nn.Softmax(dim=2))
        
        if self.encoder_type == "SP":
            out_dim = num_filters[3]*outmap_size*2
        
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3]*outmap_size*2
        
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Sequential(MaxoutLinear(out_dim, nOut), nn.BatchNorm1d(nOut, affine=False))

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        
        return out

    def forward(self, x):

        with torch.no_grad():
            
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                
                if self.log_input: x = x.log()
                
                x = self.instancenorm(x).unsqueeze(1)

        ###########################################################
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        mean_x = torch.mean(x, dim=3)
        std_x = torch.std(x, dim=3)
        x = torch.cat((mean_x, std_x), dim=2)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        # Уровень выходного слоя

        ###########################################################

        return x
        
class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))
        
class MainModel(nn.Module):

    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__()

        self.__S__ = model
        self.__L__ = trainfunc

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda() 
        outp = self.__S__.forward(data)

        if label == None:
            
            return outp

        else:
            outp = outp.reshape(1, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1
            
            
class train_dataset_loader(Dataset):
    # Train dataset loader
    
    def __init__(self, train_list, max_frames, train_path, augment=False, musan_path=None, rir_path=None):

        self.max_frames  = max_frames
        self.augment     = augment

        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in train_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(train_list):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)

        if self.augment:
            ###########################################################
            # Here is your code
            if random.random() < 0.2:  # Augment with 0.2 probability.
                if random.random() < 0.25:  # Reverberate with 0.25 probability.
                    audio = self.augment_wav.reverberate(audio)
                else:  # Add noise with 0.75 probability.
                    noisecat = random.choice(self.augment_wav.noisetypes)
                    audio = self.augment_wav.additive_noise(noisecat, audio)
            ###########################################################
            
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)
    
class test_dataset_loader(Dataset):
    # Test dataset loader
    
    def __init__(self, test_list, max_frames, test_path):

        self.max_frames  = max_frames

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in test_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(test_list):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(test_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):

        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=1)

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)

class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))