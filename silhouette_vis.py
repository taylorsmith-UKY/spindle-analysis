from __future__ import print_function

from sklearn.metrics import silhouette_samples, silhouette_score
import os

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------- PARAMETERS -----------------------------------#
base_path = 'cluster_data/'

paths = [# base_path + 'clusters/c3_freq_dist_bp/distcut/',
         # base_path + 'clusters/cvae_enc25_bp/distcut/',
         base_path + 'clusters/hyclass_pca/distcut/']

# --------------------------------------------------------------------------------#


frequencies = np.loadtxt(base_path + 'features/frequencies.txt').reshape(-1, 1)
durations = np.loadtxt(base_path + 'features/durations.txt').reshape(-1, 1)
locations = np.loadtxt(base_path + 'features/locations.txt').reshape(-1, 1)

X = np.hstack((frequencies, durations, locations))

for path in paths:
    ct = 0
    temp = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        if ct > 0:
            continue
        for dirname in dirnames:
            if dirname != 'examples':
                lbl_file = dirpath + dirname + '/clusters.txt'
                cluster_labels = np.loadtxt(lbl_file, dtype=str)
                cluster_names = np.unique(cluster_labels)
                n_clusters = len(cluster_names)
                if len(cluster_labels) != X.shape[0]:
                    continue

                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)

                # The 1st subplot is the silhouette plot
                # The silhouette coefficient can range from -1, 1 but in this example all
                # lie within [-0.1, 1]
                ax1.set_xlim([-1, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)

                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[np.where(cluster_labels == cluster_names[i])]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = plt.cm.Spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                colors = plt.cm.Spectral(np.arange(n_clusters) / n_clusters)
                ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                # Labeling the clusters
                centers = np.zeros((n_clusters, X.shape[1]))
                for i in range(n_clusters):
                    idx = cluster_names[i]
                    sel = np.where(cluster_labels == idx)[0]
                    centers[i, :] = np.mean(X[sel], axis=0)

                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for Spindle Frequency")
                ax2.set_ylabel("Feature space for Duration")

                plt.suptitle(
                    ("Silhouette analysis for distribution based hierarchical clustering on data with n_clusters = %d" %
                     n_clusters), fontsize=14, fontweight='bold')

                plt.tight_layout()
                plt.savefig(dirpath + dirname + '/silhouette.png')
                plt.show()
        ct += 1
