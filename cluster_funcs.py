#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:44:05 2018

@author: taylorsmith
"""
import matplotlib.pyplot as plt
import numpy as np
import fastcluster as fc
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist


ch_names=['Fp1-A2','Fp2-A2','F7-A2','F3-A2','Fpz-A2','F4-A2','F8-A2','T3-A2','C3-A2','Cz-A2','C4-A2','T4-A2','T5-A2','P3-A2','Pz-A2','P4-A2','T6-A2','O1-A2','O2-A2']
front = range(6)
center = range(6,11)
rear = range(11,20)

#%%
def cluster(X,ids,path,method='ward',metric='euclidean'):
    #if X is 1-D, it is condensed distance matrix, otherwise it is assumed to be
    #an array of m observations of n dimensions
    link = fc.linkage(X,method=method,metric=metric)
    dend = dendrogram(link,labels=ids,no_plot=True)
    order = np.array(dend['leaves'],dtype=int)
    c_ids = ids[order]
    np.savetxt(path+'ids_cluster_order.txt',c_ids,fmt='%d')
    np.savetxt(path+'linkage_data.txt',link)
    return link

def clust_grps(n_clusts,path):
    link = np.loadtxt(path+'linkage_data.txt')
    grps = fcluster(link,n_clusts,'maxclust')
    np.savetxt(path+str(n_clusts)+'_clusters.txt',grps,fmt='%d')
    return grps

def plot_dend(link,path):
    fig = plt.figure(figsize=(6,6))
    dendrogram(link)
    plt.xlabel('Patient ID')
    plt.ylabel('Distance')
    plt.suptitle('Cluster Dendrogram', fontweight='bold', fontsize=14)
    plt.savefig(path+'cluster_dend.png')
    plt.close(fig)
    return

def get_dist(X,outpath,metric='euclidean'):
    dist = pdist(X,metric=metric)
    np.savetxt(outpath+metric+'_dist.txt',dist)
    return dist


#%%
def cluster_stats(stats,outpath):
    all_clusters = np.loadtxt(outpath+'clusters.txt')
    n_clusts = int(np.max(all_clusters))
    clusters = []
    for i in range(n_clusts):
        clusters.append(np.where(all_clusters == i+1)[0])

    fq = np.linspace(7,42,139)

    ch_fq_amp = []
    ch_amp = []
    ton = []
    dur = []
    dom_freq = []


    for i in range(len(clusters)):
        ids = np.sort(clusters[i])

        ch_fq_dist = np.mean(stats['freq_amp_dist'][ids,:,:],axis=0)
        ch_dist = np.mean(stats['ch_amp'][ids,:],axis=0)
        tline = np.array([stats['time_of_night'][idx] for idx in ids])/256
        lens = np.array([stats['durations'][idx] for idx in ids])
        doms = stats['freq_amp_max'][ids,:]

        ch_fq_amp.append(ch_fq_dist)
        ch_amp.append(ch_dist)
        ton.append(tline)
        dur.append(lens)
        dom_freq.append(doms)

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(2,2,1)
        ax.pcolormesh(np.linspace(7,42,139)[:70],range(20),ch_fq_dist[:,:70])
        plt.yticks(np.arange(19)+0.5,ch_names,fontsize=8)
        plt.ylabel('Channel')
        plt.xlabel('Frequency (Hz)')
        plt.title('Representative Channel/Frequency\n Amplitude Profile')

        ax = plt.subplot(2,2,2)
        for j in range(len(ch_names)):
            ax.plot(fq,ch_fq_dist[j,:],label=ch_names[j])
        plt.ylabel('Amplitude (uV)')
        plt.xlabel('Frequency (Hz)')
        plt.ylim((0,3000))
        plt.legend(loc='upper right',fontsize=5)
        plt.xlim((8,25))
        plt.title('Average Frequency Distributions\n by Channel')

        ax = plt.subplot(2,2,3)
        ax.hist(lens)
        plt.xlim((0.2,2.0))
        plt.xlabel('Duration (s)')
        plt.ylabel('# of Spindles')
        plt.ylim((0,100))
        plt.title('Distribution of Spindle Duration')

        ax = plt.subplot(2,2,4)
        ax.hist(tline)
        plt.xlim((0,8.5))
        plt.xlabel('Time (hrs)')
        plt.ylabel('# of Spindles')
        plt.ylim((0,60))
        plt.title('Distribution of Spindle Timeline Occurence')

        plt.suptitle('Cluster '+str(i+1)+' - Summary Statistics')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(outpath+'cluster'+str(i+1)+'_summary.png')
        plt.close(fig)

    fig = plt.figure(figsize=(5,5))
    for i in range(len(clusters)):
        plt.hist(dom_freq[i][:,8],alpha=0.5,label=str(i+1))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('# of Spindles')
    plt.legend(loc='upper right')
    plt.title('Channel C3-A2 Dominant Frequency')
    plt.savefig(outpath+'dom_freq_overview.png')
    plt.close(fig)
