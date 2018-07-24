import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# ----------------------------- PARAMETERS ----------------------------- #
ch_names = np.array(
            ['Fp1-A2', 'Fp2-A2', 'F7-A2', 'F3-A2', 'Fpz-A2', 'F4-A2', 'F8-A2',
             'T3-A2', 'C3-A2', 'Cz-A2', 'C4-A2', 'T4-A2', 'T5-A2',
             'P3-A2', 'Pz-A2', 'P4-A2', 'T6-A2', 'O1-A2', 'O2-A2'])

# Make sure that clusters.txt is in the outpath
outpaths = ['cluster_data/clusters/hyclass_pca/distcut/', ]
dataFile = 'stats.h5'
fq_range = (8, 16)
# ----------------------------------------------------------------------- #

# Load stats for all patients
stats = h5py.File(dataFile, 'r')

for outpath in outpaths:
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(outpath):
        if dirpath == outpath:
            for dirname in dirnames:
                try:
                    if dirname != '10_clusters':
                        continue
                    tpath = outpath + dirname + '/'
                    # Load clusters and find corresponding indices
                    all_clusters = np.loadtxt(tpath + 'clusters.txt', dtype=str)
                    lbls = np.unique(all_clusters)
                    n_clusts = int(len(lbls))
                    clusters = []
                    for i in range(n_clusts):
                        clusters.append(np.where(all_clusters == lbls[i])[0])

                    # Count the number of frequency points
                    nfq = stats['c3_freq_dist'].shape[1]

                    # Corresponding values for frequency
                    fq = np.linspace(fq_range[0], fq_range[1], nfq)

                    # Open output file and write header
                    out = open(tpath + 'cluster_stats.csv', 'w')
                    out.write('Cluster_No,Count,Dominant_Freq_Avg,Dominant_Freq_Std,Channel,Duration_Avg,Duration_Std,Amplitude_Avg,Amplitude_Std,TON_Avg,TON_Std\n')

                    # Get stats for each cluster
                    for i in range(len(clusters)):
                        ids = np.sort(clusters[i])
                        count = len(ids)

                        # avg signal intensity for each channel-fq pair within the cluster
                        this_dist = stats['c3_freq_dist'][ids, :]
                        avg_fq_dist = np.mean(this_dist, axis=0)

                        # distribution of the time of night that the spindles occurred
                        tline = np.array([stats['time_of_night'][idx] for idx in ids])

                        # durations
                        lens = np.array([stats['durations'][idx] for idx in ids])
                        dur_avg = np.mean(lens)
                        dur_std = np.std(lens)

                        # dominant spindle frequency for each channel and each patient
                        dom_fq = [stats['dom_fq'][idx] for idx in ids]
                        val_fq = [x for x in dom_fq if x > 0]
                        dom_fq_avg = np.mean(val_fq)
                        dom_fq_std = np.std(val_fq)

                        # dominant channel for cluster
                        ch_amp = np.array([stats['ch_amp'][idx] for idx in ids])
                        avg_amp = np.mean(ch_amp, axis=0)
                        ch_idx = np.argsort(avg_amp)[-1]
                        ch_name = ch_names[ch_idx]

                        # avg signal intensity by channel
                        amp_avg = np.mean([np.max(ch_amp[x, :]) for x in range(count)])
                        amp_std = np.std(np.max(this_dist, axis=1))

                        ton_avg = np.mean(tline)
                        ton_std = np.std(tline)

                        fig = plt.figure(figsize=(8, 8))
                        ax = plt.subplot(2, 2, 1)
                        plt.hist(dom_fq, bins=20, range=fq_range)
                        plt.xlim(fq_range)
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Number of Spindles')
                        plt.title('Distribution of Spindle Frequencies\nChannel C3')

                        ax = plt.subplot(2, 2, 2)
                        ax.plot(fq, avg_fq_dist)
                        plt.ylabel('Aggregate Power Density (uV^2/Hz * s)')
                        plt.xlabel('Frequency (Hz)')
                        plt.xlim((8, 17))
                        plt.title('Average Frequency Distribution\nChannel C3')

                        ax = plt.subplot(2, 2, 3)
                        ax.hist(lens)
                        plt.xlim((0.2, 2.0))
                        plt.xlabel('Duration (s)')
                        plt.ylabel('# of Spindles')
                        plt.title('Distribution of\nSpindle Duration')

                        ax = plt.subplot(2, 2, 4)
                        ax.hist(tline)
                        plt.xlim((0, 8.5))
                        plt.xlabel('Time (hrs)')
                        plt.ylabel('# of Spindles')
                        plt.title('Distribution of\nSpindle Timeline Occurence')

                        plt.suptitle('Cluster ' + str(lbls[i]) + ' - Summary Statistics')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.savefig(tpath + 'cluster' + str(lbls[i]) + '_summary.png')
                        plt.close(fig)
                        out.write('%s,%d,%f,%f,%s,%f,%f,%f,%f,%f,%f\n' %
                                  (lbls[i], count, dom_fq_avg, dom_fq_std, ch_name, dur_avg, dur_std, amp_avg, amp_std, ton_avg, ton_std))
                    out.close()
                except:
                    print('Missing cluster file in directory: ' + tpath)
