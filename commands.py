import numpy as np
import h5py
from cluster_funcs import ward_breakdown
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from sklearn.metrics.cluster.unsupervised import silhouette_samples as ss
from preprocess_funcs import butter_bandpass_filter
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import squareform
import mne

ch_names = ['Fp1-A2', 'Fp2-A2', 'F7-A2', 'F3-A2', 'Fpz-A2', 'F4-A2', 'F8-A2', 'T3-A2', 'C3-A2', 'Cz-A2', 'C4-A2',
            'T4-A2', 'T5-A2', 'P3-A2', 'Pz-A2', 'P4-A2', 'T6-A2', 'O1-A2', 'O2-A2']


def get_breakdowns(h5_name, sf_name, all_inputs, lbl_paths, out_paths, thresholds):
    assert len(all_inputs) == len(lbl_paths) == len(out_paths)

    f = h5py.File(h5_name, 'r')
    sf = h5py.File(sf_name, 'r')
    dur = sf['durations'][:]
    freq = sf['dom_fq'][:]
    temp = sf['ch_amp'][:]
    loc = np.zeros(len(temp))
    for i in range(len(temp)):
        loc[i] = np.argmin(temp[i])

    for i in range(len(all_inputs)):
        if not os.path.exists(out_paths[i]):
            os.mkdir(out_paths[i])
        inputs = f[all_inputs[i]][:]
        lbls = np.loadtxt(lbl_paths[i]).astype(int)
        clbls = np.unique(lbls)
        for p_thresh in thresholds:
            try:
                for bkdn_num in clbls:
                    lbls = ward_breakdown(inputs, lbls, freq, dur, loc, bkdn_num, p_thresh=p_thresh, method='significant')
                n_clust = str(len(np.unique(lbls)))
                tpath = out_paths[i] + n_clust + 'clusters_p%.0E/' % p_thresh
                os.mkdir(tpath)
                np.savetxt(tpath + 'clusters.txt', lbls, fmt='%d')
            except:
                print(all_inputs[i] + ' was not separable with p=%.2E' % p_thresh)


def visualize_clusters(spindles, spectrograms, lbls, dm, basepath):
    if len(np.shape(dm)) == 1:
        sqdm = squareform(dm)
    else:
        sqdm = dm
    clbls = np.unique(lbls)
    n_clusters = len(clbls)
    cluster_centers = np.zeros(((n_clusters, ) + spindles.shape[1:]))
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    for i in range(n_clusters):
        tlbl = clbls[i]
        sel = np.where(lbls == tlbl)[0]
        idx = np.ix_(sel, sel)
        tdm = sqdm[idx]
        dsums = np.sum(tdm, axis=1)
        o = np.argsort(dsums)
        cluster_centers[i, :, :] = spindles[sel[o[0]]]
        ex_idx = np.concatenate((sel[o[:3]], sel[o[-3:]]))
        names = ['cluster_center', 'second_lowest_icd', 'third_lowest_icd', 'third_highest_icd',
                 'second_highest_icd', 'highest_icd']
        if not os.path.exists(basepath + str(tlbl) + '/'):
            os.makedirs(basepath + str(tlbl) + '/')
        for j in range(6):
            sig = spindles[ex_idx[j], :, :]
            spec = spectrograms[ex_idx[j], :, :, :]
            bp = np.zeros(sig.shape)
            for k in range(bp.shape[0]):
                bp[k, :] = butter_bandpass_filter(sig[k, :], 8, 16, 256)

            # Channel C3 with center channel of spectrogram
            fig = plt.figure(figsize=(3, 8))
            gs = gridspec.GridSpec(4, 1)
            fig.subplots_adjust(hspace=.4)
            plt.subplot(gs[0])
            plt.plot(np.linspace(0, 4, spindles.shape[-1]), sig[8, :])
            plt.title('Raw Signal')
            plt.xlim((0, 4))
            plt.subplot(gs[1])
            plt.plot(np.linspace(0, 4, spindles.shape[-1]), bp[8, :])
            plt.title('Band-Passed (8-16 Hz)')
            plt.xlim((0, 4))
            ax3 = plt.subplot(gs[2:])
            plt.pcolormesh(spec[:, :, 1])
            plt.title('Concatenated Spectrogram')
            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.suptitle('Cluster ' + str(tlbl) + ' ' + names[j])
            plt.savefig(basepath + str(tlbl) + '/' + names[j] + '_wspec.png')

            # All channels, signal and bp
            plt.figure(figsize=(8, 8))
            for k in range(sig.shape[0]):
                ax1 = plt.subplot(sig.shape[0], 2, 2 * k + 1)
                plt.plot(np.linspace(0, 4, sig.shape[-1]), sig[k, :])
                h = plt.ylabel(ch_names[k] + ' (uHz)')
                h.set_rotation(0)
                if k == 0:
                    ax1.set_title('Raw Signal')
                elif k < sig.shape[0] - 1:
                    ax1.set_xticklabels([])
                else:
                    ax1.set_xlabel('Time (s)')
                ax2 = plt.subplot(sig.shape[0], 2, 2 * k + 2)
                plt.plot(np.linspace(0, 4, sig.shape[-1]), bp[k, :])
                if k == 0:
                    ax2.set_title('Band-passed Signal')
                elif k < sig.shape[0] - 1:
                    ax2.set_xticklabels([])
                else:
                    ax2.set_xlabel('Time (s)')
            plt.suptitle('Cluster ' + str(tlbl) + ' ' + names[j])
            plt.savefig(basepath + str(tlbl) + '/' + names[j] + '_all_channels.png')


def get_heatmaps(sf_name, path):
    sf = h5py.File(sf_name, 'r')
    dur = sf['durations'][:]
    freq = sf['dom_fq'][:]
    temp = sf['ch_amp'][:]
    loc = np.zeros(len(temp))
    for i in range(len(temp)):
        loc[i] = np.argmin(temp[i])

    lbls = np.loadtxt(path + 'clusters.txt', dtype=int)
    n_clust = len(np.unique(lbls))

    freq_ss = ss(freq.reshape(-1, 1), lbls)
    dur_ss = ss(dur.reshape(-1, 1), lbls)
    loc_ss = ss(loc.reshape(-1, 1), lbls)

    out = np.zeros((n_clust, 3))
    for i in range(n_clust):
        idx = np.unique(lbls)[i]
        sel = np.where(lbls == idx)[0]
        out[i, 0] = np.mean(freq_ss[sel])
        out[i, 1] = np.mean(dur_ss[sel])
        out[i, 2] = np.mean(loc_ss[sel])
    np.savetxt(path + 'silhouettes.csv', out, delimiter=',', fmt='%.3f')
    return out

#
#
#
#
# base_path = 'cluster_data/'
# feature = 'c3_freq_dist'
# n_clusters = 47
# p_val = 1e-10
# parent_name = '1-1-1-2-1-2'
# text_locs = ((0.1, 0.9), (0.9, 0.9), (0.1, 0.9))
#
# frequencies = np.loadtxt(base_path + 'features/frequencies.txt').reshape(-1, 1)
# durations = np.loadtxt(base_path + 'features/durations.txt').reshape(-1, 1)
# locations = np.loadtxt(base_path + 'features/locations.txt').reshape(-1, 1)
# fq_dm = squareform(pdist(frequencies, metric='cityblock'))
# dur_dm = squareform(pdist(durations, metric='cityblock'))
# loc_dm = squareform(pdist(locations, metric='cityblock'))
# path = base_path + 'clusters/' + feature + '/uniform/dyncut/%dclusters_p%.0E/' % (n_clusters, p_val)
# cluster_labels = np.loadtxt(path + 'clusters.txt', dtype=str)
# label_names = np.unique(cluster_labels)
# left_name = parent_name + '-1'
# right_name = parent_name + '-2'
# left_idx = np.where(cluster_labels == left_name)[0]
# right_idx = np.where(cluster_labels == right_name)[0]
# parent_idx = np.union1d(left_idx, right_idx)
# left_intra_idx = np.ix_(left_idx, left_idx)
# left_inter_loc_idx = np.ix_(left_idx, right_idx)
# left_inter_glob_idx = np.ix_(left_idx, np.setdiff1d(np.arange(len(features)), left_idx))
# right_intra_idx = np.ix_(right_idx, right_idx)
# right_inter_loc_idx = np.ix_(right_idx, left_idx)
# right_inter_glob_idx = np.ix_(right_idx, np.setdiff1d(np.arange(len(features)), right_idx))
# left_frequencies = frequencies[left_idx]
# left_durations = durations[left_idx]
# left_locations = locations[left_idx]
# right_frequencies = frequencies[right_idx]
# right_durations = durations[right_idx]
# right_locations = locations[right_idx]
#
# all_cluster_features = np.loadtxt(base_path + 'features/cvae_enc25.txt')
#
#
# left_cluster_features = all_cluster_features[left_idx]
# right_cluster_features = all_cluster_features[right_idx]


def plot_multi_distributions(all_feats, all_idx, feat_names=[], grp_names=[], arg_dic={}, fig_types=['hist'], save_path='', text_locs = []):
    n_feats = len(all_feats)
    n_grps = len(all_idx)
    n_figs = len(fig_types)
    if feat_names:
        assert len(feat_names) == n_feats
    else:
        feat_names = ['feature_%d' % x for x in range(n_feats)]

    if 'figsize' in list(arg_dic):
        figsize = arg_dic['figsize']
    else:
        figsize = (10, 4)
    if 'hist' in fig_types:
        if 'mean_line_alignments' in list(arg_dic):
            ml_al = arg_dic['mean_line_alignments']
        else:
            ml_al = [None for x in range(n_feats)]
        if 'nbins' in list(arg_dic):
            nbins = arg_dic['nbins']
        else:
            nbins = None

    if grp_names:
        assert len(all_idx) == len(grp_names)
    else:
        grp_names = [None for x in range(n_grps)]

    fig = plt.figure(figsize=figsize)
    for i in range(n_figs):
        if fig_types[i] == 'hist':
            for j in range(n_feats):
                ax = fig.add_subplot(n_figs, n_feats, i + j + 1)
                for k in range(n_grps):
                    tfeats = all_feats[j][all_idx[k]]
                    if n_grps == 2:
                        t2feats = all_feats[j][all_idx[1]]
                        tmean = np.mean(tfeats)
                        t2mean = np.mean(t2feats)
                        if tmean < t2mean:
                            if k == 0:
                                al = 'left'
                            else:
                                al = 'right'
                        else:
                            if k == 0:
                                al = 'right'
                            else:
                                al = 'left'
                    else:
                        al = 'left'
                    h = plt.hist(tfeats, bins=nbins, alpha=0.6, label=grp_names[k])
                    plt.axvline(np.mean(tfeats))
                    if al == 'right':
                        plt.text(np.mean(tfeats) + 0.01, max(h[0]) * 0.6, grp_names[k] + ' mean', rotation=90)
                    else:
                        plt.text(np.mean(tfeats) - 0.045, max(h[0]) * 0.6, grp_names[k] + ' mean', rotation=90)
                if n_grps == 2:
                    _, p_val = ttest_ind(all_feats[j][all_idx[0]], all_feats[j][all_idx[1]])
                    plt.text(text_locs[j][0], text_locs[j][1], 'p-value=%.2E' % p_val, transform=ax.transAxes)
                if feat_names and i == 0:
                    plt.title(feat_names[j])
                    plt.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path + 'split_silhouette.png')
    plt.show()
    return

