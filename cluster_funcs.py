#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:44:05 2018

@author: taylorsmith
"""
import matplotlib.pyplot as plt
import numpy as np
import fastcluster as fc
from scipy.cluster.hierarchy import dendrogram, fcluster, ward
from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.stats.mstats import normaltest
from scipy.stats import ttest_ind
from dtw import dtw
from sklearn.metrics import silhouette_samples
import os

ch_names = ['Fp1-A2', 'Fp2-A2', 'F7-A2', 'F3-A2', 'Fpz-A2', 'F4-A2', 'F8-A2', 'T3-A2', 'C3-A2', 'Cz-A2', 'C4-A2',
            'T4-A2', 'T5-A2', 'P3-A2', 'Pz-A2', 'P4-A2', 'T6-A2', 'O1-A2', 'O2-A2']
front = range(6)
center = range(6, 11)
rear = range(11, 20)


# %%
def cluster(X,ids,path,method='ward',metric='euclidean'):
    # if X is 1-D, it is condensed distance matrix, otherwise it is assumed to be
    # an array of m observations of n dimensions
    link = fc.linkage(X,method=method,metric=metric)
    dend = dendrogram(link,labels=ids,no_plot=True)
    plt.savefig(path+'dendrogram.png')
    order = np.array(dend['leaves'],dtype=int)
    c_ids = ids[order]
    np.savetxt(path+'ids_cluster_order.txt',c_ids,fmt='%s')
    np.savetxt(path+'linkage_data.txt',link)
    return link

def clust_grps(link,n_clusts,path):
    grps = fcluster(link,n_clusts,'maxclust')
    np.savetxt(path+'clusters.txt',grps,fmt='%d')
    return grps

def plot_dend(link,ids,path):
    fig = plt.figure(figsize=(6,6))
    dendrogram(link,labels=ids,color_threshold=7.5)
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

    fq = np.linspace(8, 16, 31)

    ch_amp = []
    ton = []
    dur = []
    spindle_freq_dist = []

    out = open(outpath+'cluster_stats.csv','w')
    out.write('Cluster_No,Count,Dominant_Freq_Avg,Dominant_Freq_Std,Channel,Duration_Avg,Duration_Std,Amplitude_Avg,Amplitude_Std,TON_Avg,TON_Std\n')

    for i in range(len(clusters)):
        ids = np.sort(clusters[i])
        count = len(ids)

        this_dist = stats['c3_freq_dist'][ids, :]

        # avg signal intensity for each channel-fq pair within the cluster
        ch_fq_dist = np.mean(this_dist, axis=0)
        # ch_avg = np.mean(ch_fq_dist, axis=1)
        spindle_freq_dist.append([stats['dom_fq2'][idx] for idx in ids])

        # distribution of the time of night that the spindles occurred
        tline = np.array([stats['time_of_night'][idx] for idx in ids])

        # durations
        lens = np.array([stats['durations'][idx] for idx in ids])
        dur_avg = np.mean(lens)
        dur_std = np.std(lens)

        # peak spindle frequency for each channel and each patient
        dom_fq = [stats['dom_fq2'][idx] for idx in ids]
        val_fq = [x for x in dom_fq if x > 0]
        dom_fq_avg = np.mean(val_fq)
        dom_fq_std = np.std(val_fq)

        # determine which channel has the greatest spindle intensity for this cluster
        # ch_idx = np.argsort(ch_avg)[-1]
        # ch_name = ch_names[ch_idx]

        #avg signal intensity by channel
        avg_amp = np.mean(np.max(this_dist, axis=1))
        avg_amp_std = np.std(np.max(this_dist, axis=1))

        ton_avg = np.mean(tline)
        ton_std = np.std(tline)
        ton.append(tline)
        dur.append(lens)

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(2,2,1)
        plt.hist(dom_fq, bins=20, range=(8, 16))
        plt.xlim((8,16))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Number of Spindles')
        plt.title('Distribution of Spindle Frequencies')
        # ax.pcolormesh(np.linspace(8, 17, 36), np.arange(19), ch_fq_dist[:, :])
        # plt.yticks(np.arange(19)+0.5, ch_names, fontsize=8)
        # plt.ylabel('Channel')
        # plt.xlabel('Frequency (Hz)')
        # plt.title('Representative Channel/Frequency\n Amplitude Profile')

        ax = plt.subplot(2,2,2)
        # for j in range(len(ch_names)):
        ax.plot(fq,ch_fq_dist)
        plt.ylabel('Aggregate Power Density (uV^2/Hz * s)')
        plt.xlabel('Frequency (Hz)')
        # plt.ylim((0,1400))
        # plt.legend(loc='upper right',fontsize=5)
        plt.xlim((8,17))
        plt.title('Average Frequency Distributions\n Channel C3')

        ax = plt.subplot(2,2,3)
        ax.hist(lens)
        plt.xlim((0.2,2.0))
        plt.xlabel('Duration (s)')
        plt.ylabel('# of Spindles')
        # plt.ylim((0,2000))
        plt.title('Distribution of\nSpindle Duration')

        ax = plt.subplot(2,2,4)
        ax.hist(tline)
        plt.xlim((0,8.5))
        plt.xlabel('Time (hrs)')
        plt.ylabel('# of Spindles')
        # plt.ylim((0,600))
        plt.title('Distribution of\nSpindle Timeline Occurence')

        plt.suptitle('Cluster '+str(i+1)+' - Summary Statistics')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(outpath+'cluster'+str(i+1)+'_summary.png')
        plt.close(fig)
        out.write('%d,%d,%f,%f,C3,%f,%f,%f,%f,%f,%f\n' %
                  (i, count, dom_fq_avg, dom_fq_std, dur_avg, dur_std, avg_amp, avg_amp_std, ton_avg, ton_std))

    #fig = plt.figure(figsize=(5,5))
    for i in range(len(clusters)):
        try:
            plt.figure()
            plt.hist(spindle_freq_dist[i],bins=30,label=str(i+1))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('# of Spindles')
            # plt.legend(loc='upper right')
            plt.xlim((8, 13))
            plt.title('Dominant Channel Frequency Distribution\nCluster'+str(i+1))
            plt.savefig(outpath+'fq_dist_cluster'+str(i+1)+'.png')
            plt.close(fig)
        except:
            print spindle_freq_dist[i][0]


def eval_clusters(inputs, clusters, dist_func='euclidean', fig_path='', op='max', plot='both'):
    dm = pdist(inputs, metric=dist_func)
    sqdm = squareform(dm)
    lbls = np.unique(clusters)
    nclust = len(lbls)
    ss = silhouette_samples(sqdm, clusters, metric='precomputed')
    # Normalized mutual info and other scores
    #############
    print('Global Silhouette Score: %.4f' % np.mean(ss))
    for i in range(nclust):
        # Normalized mutual info and other scores for each cluster after breakdown
        ################
        print('Cluster %d Silhouette Score: %.4f' % (lbls[i], np.mean(ss[np.where(clusters == lbls[i])])))
    inter_intra_dist(sqdm, clusters, op=op, out_path=fig_path, plot=plot)
    return ss


def inter_intra_dist(indm, clusters, op='max', out_path='', plot='both'):
    sqdm = np.array(indm, copy=True)
    if sqdm.ndim == 1:
        sqdm = squareform(sqdm)

    np.fill_diagonal(sqdm, np.nan)

    lbls = np.unique(clusters)
    n_clusters = len(lbls)

    if op == 'mean':
        func = lambda x: np.nanmean(x)
    elif op == 'max':
        func = lambda x: np.nanmax(x)
    elif op == 'min':
        func = lambda x: np.nanmin(x)

    all_inter = []
    all_intra = []
    for i in range(sqdm.shape[0]):
        clust = clusters[i]
        idx = np.array((i,))
        cids = np.where(clusters == clust)[0]
        intra = np.ix_(idx, np.setdiff1d(cids, idx))
        inter = np.ix_(idx, np.setdiff1d(range(sqdm.shape[0]), cids))
        intrad = func(sqdm[intra])
        interd = func(sqdm[inter])
        all_inter.append(interd)
        all_intra.append(intrad)
    all_inter = np.array(all_inter)
    all_intra = np.array(all_intra)
    for i in range(n_clusters):
        clust = lbls[i]
        cidx = np.where(clusters == clust)[0]
        plt.figure()
        if plot == 'both':
            plt.subplot(121)
        if plot == 'both' or plot == 'inter':
            plt.hist(all_inter[cidx], bins=30)
            plt.title('Inter-Cluster')
        if plot == 'both':
            plt.subplot(122)
        if plot == 'both' or plot == 'intra':
            plt.hist(all_intra[cidx], bins=30)
            plt.title('Intra-Cluster')
        if plot == 'nosave':
            plt.subplot(121)
            plt.hist(all_inter[cidx], bins=30)
            plt.title('Inter-Cluster')
            plt.subplot(122)
            plt.hist(all_intra[cidx], bins=30)
            plt.title('Intra-Cluster')
            plt.suptitle('Cluster ' + str(clust) + ' Separation')
            plt.show()
        if plot == 'both':
            if clust >= 0:
                plt.suptitle('Cluster ' + str(clust) + ' Separation')
                plt.savefig(out_path+'cluster' + str(clust) + '_separation_hist.png')
            else:
                plt.suptitle(out_path+'Noise Point Separation')
                plt.savefig(out_path+'noise_separation_hist.png')
        elif plot == 'inter':
            plt.savefig(out_path + 'cluster' + str(clust) + '_inter_dist_hist.png')
        elif plot == 'intra':
            plt.savefig(out_path + 'cluster' + str(clust) + '_intra_dist_hist.png')
        plt.close()

    np.savetxt(out_path+'inter_intra_dist.txt', np.transpose(np.vstack((all_inter, all_intra))))
    return sqdm, all_inter, all_intra


def ward_breakdown(inputs, lbls, freq, dur, loc, bkdn_num, p_thresh=1e-3, metric='euclidean', min_size=15):
    sel = np.where(lbls == bkdn_num)[0]
    if len(sel) < min_size:
        return lbls
    ds = inputs[sel]
    tfreq = freq[sel]
    tdur = dur[sel]
    tloc = loc[sel]

    print(method)
    if method == 'normal':
        _, p_f = normaltest(tfreq)
        _, p_d = normaltest(tdur)
        _, p_l = normaltest(tloc)
        print('%s, %s, %s' % (str(p_f), str(p_d), str(p_l)))
        if np.all(np.array((p_f, p_d, p_l)) < p_thresh):
            return lbls

    max_id = np.max(lbls)

    condensed = pdist(ds, metric=metric)

    link = ward(condensed)

    nlbls = fcluster(link, 2, criterion='maxclust')
    sel1 = np.where(nlbls == 1)[0]
    sel2 = np.where(nlbls == 2)[0]

    if len(sel1) < min_size or len(sel2) < min_size:
        return lbls

    freq1 = tfreq[sel1]
    dur1 = tdur[sel1]
    loc1 = tloc[sel1]

    freq2 = tfreq[sel2]
    dur2 = tdur[sel2]
    loc2 = tloc[sel2]

    if method == 'normal':
        lbls[sel1] = max_id + 1
        lbls[sel2] = max_id + 2
        max_id += 2
        if len(sel1) > 20:
            _, p_f = normaltest(freq1)
            _, p_d = normaltest(dur1)
            _, p_l = normaltest(loc1)
            if np.any(np.array((p_f, p_d, p_l)) > p_thresh):
                print('Split Node: %d to [%d, %d]' % (max_id - 1, max_id))
                lbls = ward_breakdown(inputs, lbls, freq, dur, loc, max_id - 1, p_thresh, metric, method, min_size)

        if len(sel2) > 20:
            _, p_f = normaltest(freq2)
            _, p_d = normaltest(dur2)
            _, p_l = normaltest(loc2)
            if np.any(np.array((p_f, p_d, p_l)) > p_thresh):
                lbls = ward_breakdown(inputs, lbls, freq, dur, loc, max_id, p_thresh, metric, method, min_size)
    elif method == 'significant':
        _, p_f = ttest_ind(freq1, freq2)
        _, p_d = ttest_ind(dur1, dur2)
        _, p_l = ttest_ind(loc1, loc2)
        # print('%s, %s, %s' % (str(p_f), str(p_d), str(p_l)))
        if np.any(np.array((p_f, p_d, p_l)) < p_thresh):
            print('Split Node: %d to [%d, %d]' % (bkdn_num, max_id + 1, max_id + 2))
            lbls[sel[sel1]] = max_id + 1
            lbls[sel[sel2]] = max_id + 2
            max_id += 2
            lbls = ward_breakdown(inputs, lbls, freq, dur, loc, max_id - 1, p_thresh, metric, method, min_size)
            lbls = ward_breakdown(inputs, lbls, freq, dur, loc, max_id, p_thresh, metric, method, min_size)

    return lbls


def pairwise_dtw_dist(ds, metric=euclidean):
    dist = []
    for i in range(len(ds)):
        for j in range(i+1, len(ds)):
            d, _, _, path = dtw(ds[i, :], ds[j, :], metric)
            p1_path = path[0]
            p2_path = path[1]
            p1 = [ds[i][p1_path[x]] for x in range(len(p1_path))]
            p2 = [ds[j][p2_path[x]] for x in range(len(p2_path))]
            dist.append(metric(p1, p2))
    return dist
