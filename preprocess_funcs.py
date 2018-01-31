from __future__ import division
import h5py
import numpy as np
import pyedflib
import re
import os
from scipy.stats import entropy
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
#from sklearn.decomposition import PCA
import scipy.ndimage as ndi
from scipy.signal import argrelextrema as rel_max

ch_names=['Fp1-A2','Fp2-A2','F7-A2','F3-A2','Fpz-A2','F4-A2','F8-A2','T3-A2','C3-A2','Cz-A2','C4-A2','T4-A2','T5-A2','P3-A2','Pz-A2','P4-A2','T6-A2','O1-A2','O2-A2']

#%% Extract spindles from MASS dataset
#   specifically designed for this dataset... for others modify chan_list
#   and determine appropriate length for 3rd dimension size of ds
def get_spindles(sigfiles,detfiles,fname='spindles.h5',lbl_type='intersect'):
    #Get paths and filenames from conf
    if os.path.exists(fname):
        print("File exists. Provide new name or remove existing file: "+fname+'\n')
        exit()

    outFile=h5py.File(fname,'w')
    ds = outFile.create_dataset('spindles_temp',data=np.zeros([16000,19,1024]))

    count=0
    dur = np.array([],dtype=float)
    ton = np.array([],dtype=float)
    for i in range(len(sigfiles)):
        print('On example #'+str(i+1))
        inFile = pyedflib.EdfReader(sigfiles[i])
        chan_list=[['Fp1',re.compile('EEG Fp1-CLE')],
               ['Fp2',re.compile('EEG Fp2-CLE')],
               ['F7',re.compile('EEG F7-CLE')],
               ['F3',re.compile('EEG F3-CLE')],
               ['Fpz',re.compile('EEG Fpz-CLE')],
               ['F4',re.compile('EEG F4-CLE')],
               ['F8',re.compile('EEG F8-CLE')],
               ['T3',re.compile('EEG T3-CLE')],
               ['C3',re.compile('EEG C3-CLE')],
               ['Cz',re.compile('EEG Cz-CLE')],
               ['C4',re.compile('EEG C4-CLE')],
               ['T4',re.compile('EEG T4-CLE')],
               ['T5',re.compile('EEG T5-CLE')],
               ['P3',re.compile('EEG P3-CLE')],
               ['Pz',re.compile('EEG *[Pp]z-CLE')],
               ['P4',re.compile('EEG P4-CLE')],
               ['T6',re.compile('EEG T6-CLE')],
               ['O1',re.compile('EEG O1-CLE')],
               ['O2',re.compile('EEG O2-CLE')],
               ['A2',re.compile('EEG A.-CLE')]]
        ch_idx = np.zeros(len(chan_list)-1,dtype=int)
        all_chan=inFile.getSignalLabels()
        for j in range(len(chan_list)-1):
            ch_idx[j] = [x for x in range(len(all_chan)) if chan_list[j][1].match(all_chan[x])][0]
        comp_idx = [x for x in range(len(all_chan)) if chan_list[-1][1].match(all_chan[x])][0]

        fs=int(inFile.getSignalHeader(ch_idx[0])['sample_rate'])



        if lbl_type == 'single':
            df=pyedflib.EdfReader(detfiles[i])
            annot=df.readAnnotations()
            for j in range(len(annot[0])):
                if annot[1][j] > 10:
                    continue
                start=int(annot[0][j]*fs)
                n=int(annot[1][j]*fs)
                mid = start + int(n/2)
                wstart = mid-512
                comp = inFile.readSignal(comp_idx,wstart,1024)
                for ch in range(len(ch_idx)):
                    ds[count,ch,:]=inFile.readSignal(ch_idx[ch],wstart,1024) - comp
                count+=1
        else:
            n_files = len(detfiles[i])
            sig_len = inFile.getFileDuration()*fs
            mask = np.zeros((n_files,sig_len),dtype=bool)
            tstarts = []
            for j in range(n_files):
                df=pyedflib.EdfReader(detfiles[i][j])
                annot=df.readAnnotations()
                for k in range(len(annot[0])):
                    if annot[1][i] > 10:
                        continue
                    start=int(annot[0][k]*fs)
                    tstarts.append(annot[0][k])
                    stop=int(start+annot[1][k]*fs)
                    mask[j,start:stop]=1
            if n_files == 1:
                spindles = mask
            elif lbl_type == 'intersect':
                spindles = mask[0,:]
                for j in range(1,n_files):
                    spindles = np.logical_and(spindles,mask[j,:])
            elif lbl_type == 'union':
                spindles = mask[0,:]
                for j in range(1,n_files):
                    spindles = np.logical_or(spindles,mask[j,:])

            spindles = spindles.astype(int)
            edges = spindles[1:]-spindles[:-1]
            starts = np.where(edges==1)[0]+1
            ton=np.concatenate((ton,np.array(starts,dtype=float)/(60*60*fs)))
            '''
            dlist = []
            for j in range(1,len(starts)):
            j = 1
            while j < len(starts):
                if starts[j] - starts[j-1] < SEPARATION_THRESHOLD:
                    np.delete(starts, j)
                    dlist.append(j)
                else:
                    j+=1
            stops = np.where(edges==-1)[0]
            for idx in dlist:
                np.delete(stops,idx)
            lens = stops - starts + 1
            '''
            lens = np.where(edges==-1)[0] - starts + 1
            dur=np.concatenate((dur,np.array([float(lens[i])/fs for i in range(len(lens))])))
            for j in range(len(starts)):
                comp = inFile.readSignal(comp_idx,starts[j],lens[j])
                for ch in range(len(ch_idx)):
                    ds[count,ch,:lens[j]]=inFile.readSignal(ch_idx[ch],starts[j],lens[j]) - comp
                count+=1
    final = outFile.create_dataset('spindles',data=ds[:count,:,:])
    stats = outFile.create_group('spindle_stats')
    stats.create_dataset('durations',data=np.squeeze(dur))
    stats.create_dataset('time_of_night',data=np.squeeze(ton))
    outFile.__delitem__('spindles_temp')
    return final

#%%
def fft_fe(ds,nfft,ham_win,f_crop=(7,42),ds_name='spectrograms'):
    print('Extracting spectrogram features')
    nwin=ds.shape[0]
    nchan=ds.shape[1]
    step=ham_win/2

    fs = 256
    #Determine spectrogram dimensions and create dataset
    fq,t,Sxx = signal.spectrogram(ds[0,0,:],window=signal.get_window('hamming',ham_win),noverlap=step,nfft=nfft,detrend=None,fs=fs)
    #mask to crop extraneous frequencies
    keep = np.ones(len(fq),dtype=bool)
    idx = np.where(fq <= f_crop[0])[0]
    keep[idx] = 0
    idx = np.where(fq >= f_crop[1])[0]
    keep[idx] = 0
    nf = len(np.where(keep)[0])
    f = ds.parent

    specs = f.create_dataset(ds_name,shape=(nwin,nchan,nf,len(t)),dtype=float)
    for i in range(nwin):
        if i % 50 == 0:
            print('Window #'+str(i+1)+' of '+str(nwin))
        for ch in range(nchan):
            fq,t,Sxx=signal.spectrogram(ds[i,ch,:],window=signal.get_window('hamming',ham_win),noverlap=step,nfft=nfft,detrend=None)
            specs[i,ch,:,:]=Sxx[keep,:]
    return specs


#%%
def spindle_stats(specs,f_range=(7,42),stat_name='spindle_stats'):
    df = specs.parent
    n_specs = specs.shape[0]
    n_chan = specs.shape[1]
    n_fq = specs.shape[2]

    grp = df[stat_name]
    f_amp_dist = grp.create_dataset('freq_amp_dist',data=np.zeros([n_specs,n_chan,n_fq]))
    f_amp_max = grp.create_dataset('freq_amp_max',data=np.zeros([n_specs,n_chan]))
    ch_amp = grp.create_dataset('ch_amp',data=np.zeros([n_specs,n_chan]))
    for i in range(n_specs):
        if i % 50 == 0:
            print('On example #'+str(i+1))
        f_amp_dist[i,:,:] = np.mean(specs[i,:,:,:],axis=2)    #avg amplitude for each frequency bin
        ch_amp[i,:] = np.mean(f_amp_dist[i,:,:],axis=1)      #avg total amplitude for each channel
        for j in range(n_chan):
            try:
                max_idx = rel_max(f_amp_dist[i,j,:],np.greater)[0][0]
                f_amp_max[i,j] = np.linspace(f_range[0],f_range[1],n_fq)[max_idx]
            except:
                f_amp_max[i,j] = 0
    return grp

#%%
def normalize(inds,scale=2,v=False):
    print('Normalizing extracted features')
    f = inds.parent
    nchan=inds.shape[1]
    base_name = str(inds.name)

    norm=f.create_dataset(base_name+'_norm',shape=inds.shape,dtype=float)
    for i in range(nchan):
        print('Normalizing channel '+str(i+1)+' of '+str(nchan))
        mean=np.mean(inds[:,i,:,:])
        std=np.std(inds[:,i,:,:])
        maxim=mean+scale*std
        minim=mean-scale*std
        if minim < 0:
            minim = 0
        norm[:,i,:,:]=np.clip((inds[:,i,:,:]-minim)/(maxim-minim),a_min=0.0,a_max=1.0)
    if v:
        plt.figure()
        plt.hist(norm[i,:,:,:].flatten(),bins=30,range=(0,1))
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.title('Channel '+str(ch_names[i])+' Spectrogram Amplitude Distribution')
        plt.xlim(0,1)
        plt.savefig('spec_hist_norm_ch'+str(ch_names[i])+'.pdf')
        plt.clf()
    return norm

def resize_specs(inspecs,scale_factors=(1,3)):
    f=inspecs.parent
    print('Resizing spectrograms...')
    base_name = inspecs.name
    ds_name = base_name + '_rsz'
    height = inspecs.shape[2]*scale_factors[0]
    width = inspecs.shape[3]*scale_factors[1]

    rds = f.create_dataset(ds_name,shape=(inspecs.shape[0],inspecs.shape[1],height,width))
    for i in range(inspecs.shape[0]):
        if i % 50 == 0:
            print('Window #'+str(i+1)+' of '+str(inspecs.shape[0]))
        for j in range(inspecs.shape[1]):
            rds[i,j,:,:] = ndi.zoom(inspecs[i,j,:,:],scale_factors)
    return rds


#%%
def concat_specs(inspecs):
    f=inspecs.parent
    print('Concatenating spectrograms...')
    base_name = inspecs.name
    ds_name = base_name + '_concat'
    cds = f.create_dataset(ds_name,shape=(inspecs.shape[0],299,299,3))
    height = inspecs.shape[2]
    width = inspecs.shape[3]

    gap = int((299-(width*3))/2)
    c2_start = width+gap
    c2_stop = c2_start+width

    for i in range(inspecs.shape[0]):
        if i % 50 == 0:
            print('Window #'+str(i+1)+' of '+str(inspecs.shape[0]))
        s = np.zeros((299,299))
        #s[:]=np.inf
        #front channels
        s[0:height,0:width] = inspecs[i,3,:,:]
        s[0:height,c2_start:c2_stop] = inspecs[i,9,:,:]
        s[0:height,-width:] = inspecs[i,6,:,:]
        s[-height:,0:width] = inspecs[i,2,:,:]
        s[-height:,c2_start:c2_stop] = inspecs[i,4,:,:]
        s[-height:,-width:] = inspecs[i,5,:,:]
        cds[i,:,:,0]=s

        s = np.zeros((299,299))
        #s[:]=np.inf
        #central channels
        s[0:height,0:width] = inspecs[i,8,:,:]
        s[0:height,c2_start:c2_stop] = inspecs[i,9,:,:]
        s[0:height,-width:] = inspecs[i,10,:,:]
        s[-height:,0:width] = inspecs[i,7,:,:]
        s[-height:,c2_start:c2_stop] = inspecs[i,9,:,:]
        s[-height:,-width:] = inspecs[i,11,:,:]
        cds[i,:,:,1]=s

        s = np.zeros((299,299))
        #s[:]=np.inf
        #rear channels
        s[0:height,0:width] = inspecs[i,12,:,:]
        s[0:height,c2_start:c2_stop] = inspecs[i,18,:,:]
        s[0:height,-width:] = inspecs[i,16,:,:]
        s[-height:,0:96] = inspecs[i,13,:,:]
        s[-height:,c2_start:c2_stop] = inspecs[i,17,:,:]
        s[-height:,-width:] = inspecs[i,15,:,:]
        cds[i,:,:,2]=s
    return cds


#%%
def plot_concat(concat,idx,ind_h,ind_w,fq,t):
    tot_h = concat.shape[1]
    tot_w = concat.shape[2]

    t_gaps = tot_w - 3*ind_w
    t_gap1 = int(t_gaps/2) + t_gaps%2
    t_gap2 = int(t_gaps/2)

    t1_idx = np.linspace(0,96,5,dtype=int)
    t2_idx = t1_idx + ind_w + t_gap1
    t3_idx = t2_idx + ind_w + t_gap2


    t_idx = np.concatenate([t1_idx,t2_idx,t3_idx])
    t_lbls = tuple(np.tile(np.array(('0','1','2','3','4')),3))

    f_gap = tot_h - 2*ind_h
    f1_idx = np.linspace(0,139,6,dtype=int)
    f2_idx = f1_idx + ind_h + f_gap

    f_idx = np.concatenate([f1_idx,f2_idx])
    f_lbls = np.tile(np.linspace(7,42,6,dtype=str),2)

    #all_f = np.concatenate((ind_f,np.zeros(f_gap),ind_f))

    #all_t = np.concatenate((ind_t,np.zeros(t_gap1),ind_t,np.zeros(t_gap2),ind_t))

    front = concat[idx,:,:,0]
    middle = concat[idx,:,:,1]
    rear = concat[idx,:,:,2]
    plt.figure(figsize=(9,3),dpi=600)
    mpl.rcParams.update({'axes.linewidth': 0.2,
                         'font.size': 5,
                         'xtick.major.width': 0.2,
                         'ytick.major.width': 0.2})

    plt.subplot(1,3,1)
    plt.pcolormesh(front,vmin=0,vmax=1)
    plt.xticks(t_idx,t_lbls)
    plt.yticks(f_idx,f_lbls)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    #ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title('Front Channels')

    plt.subplot(1,3,2)
    plt.pcolormesh(middle,vmin=0,vmax=1)
    plt.xticks(t_idx,t_lbls)
    plt.yticks([],[])
    plt.xlabel('Time (s)')
    #ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title('Central Channels')

    plt.subplot(1,3,3)
    plt.pcolormesh(rear,vmin=0,vmax=1)
    plt.xticks(t_idx,t_lbls)
    plt.yticks([],[])
    plt.xlabel('Time (s)')
    #ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title('Rear Channels')
    plt.colorbar()

    plt.suptitle('Concatenated Spectrograms - Example #' + str(idx))

    plt.savefig('concat_winno_'+str(idx)+'.png')


#%%
def flatten_feats(inds):
    f=inds.parent
    ds=f.create_dataset(inds.name+'_flat',shape=(inds.shape[0],np.prod(inds.shape[1:])),dtype=float)
    for i in range(inds.shape[0]):
        ds[i,:]=inds[i,:,:,:].flatten()
    return ds


#%%
def calc_entropy(f):
    #From scipy.org:
    #entropy = S = -sum(pk * log(pk), axis=0)
    print('Calculating entropy for each feature')
    feat=f['spectrograms_norm3']
    nchan=feat.shape[1]
    nfq=feat.shape[2]
    ntpts=feat.shape[3]

    en=f.create_dataset('feat_entropy',shape=(nchan,nfq,ntpts),dtype=float)
    means=f.create_dataset('feat_means',shape=(nchan,nfq,ntpts),dtype=float)
    for ch_idx in range(nchan):
        for fq_idx in range(nfq):
            for tpt_idx in range(ntpts):
                en[ch_idx,fq_idx,tpt_idx] = entropy(feat[:,ch_idx,fq_idx,tpt_idx].flatten())
                means[ch_idx,fq_idx,tpt_idx] = np.mean(feat[:,ch_idx,fq_idx,tpt_idx].flatten())
    return en, means

#%%
def feat2csv(inds,entropy,means,nm=0.5,ne=9.3):
    outFile=open('spec_features_m05_elt093.csv','w')
    m=means[:].flatten()
    m_i = np.where(m > nm)[0]
    keepm = np.zeros(inds.shape[1],dtype=bool)
    keepm[m_i]=1
    f_importance = entropy[:].flatten()
    f_importance = f_importance[m_i]
    f_idxs = np.where(f_importance < ne)[0]
    keepe = np.zeros(inds.shape[1],dtype=bool)
    keepe[f_idxs] = 1
    feats = inds[:,keepm]
    feats = feats[:,keepe]
    nfeats = len(f_idxs)
    for i in range(nfeats):
        outFile.write(', %d' % (i+1))
    outFile.write('\n')
    for i in range(feats.shape[0]):
        outFile.write('%d' % (i+1))
        for j in range(feats.shape[1]):
            outFile.write(', %f' % (feats[i,j]))
        outFile.write('\n')
    outFile.close()

def xtrct_ds(df_name,ds_name,out_fname,out_dsname):
    inf = h5py.File(df_name,'r')
    out = h5py.File(out_fname,'w')
    out.create_dataset(out_dsname,data=inf[ds_name])
    return