import numpy as np
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import fcluster, to_tree
from scipy.spatial.distance import pdist
from fastcluster import ward
import os

# --------------------------------- PARAMETERS -----------------------------------#
# directory containing all different types of features as individual text files
base_path = 'cluster_data/'

# features used for distance calc to determine dendrogram structure
# options are [c3_freq_dist, c3_freq_dist_bp, cvae_enc25, cvae_enc25_bp, hyclass_pca]
feature_names = ['hyclass_pca', ]
# feature_names = ['cvae_enc25_bp', ]
# threshold for determining significantly different subsets
thresholds = [1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100, 1e-125, 1e-150, 1e-200, 1e-250]
min_size = 20
height_lim = 4
# --------------------------------------------------------------------------------#

frequencies = np.loadtxt(base_path + 'features/frequencies.txt').reshape(-1, 1)
durations = np.loadtxt(base_path + 'features/durations.txt').reshape(-1, 1)
locations = np.loadtxt(base_path + 'features/locations.txt').reshape(-1, 1)

all_feats = [frequencies, durations, locations]


def dist_cut_tree(node, lbls, base_name, all_feats, p_thresh, min_size=20, height_lim=5):
    height = len(base_name.split('-'))
    if height > height_lim:
        print('Height limit reached for node: %s' % base_name)
        return lbls
    left = node.get_left()
    right = node.get_right()
    left_idx = left.pre_order()
    right_idx = right.pre_order()
    split = False
    for feat in all_feats:
        _, p_val = ttest_ind(feat[left_idx], feat[right_idx])
        if p_val < p_thresh:
            split = True
    if split:
        left_name = base_name + '-l'
        right_name = base_name + '-r'
        lbls[left_idx] = base_name + '-l'
        lbls[right_idx] = base_name + '-r'
        if len(left_idx) < min_size:
            print('Node %s minimum size' % left_name)
        else:
            print('Splitting node %s' % left_name)
            lbls = dist_cut_tree(left, lbls, left_name, all_feats, p_thresh, min_size=min_size, height_lim=height_lim)

        if len(right_idx) < min_size:
            print('Node %s minimum size' % right_name)
        else:
            print('Splitting node %s' % right_name)
            lbls = dist_cut_tree(right, lbls, right_name, all_feats, p_thresh, min_size=min_size, height_lim=height_lim)
    return lbls


n_clusters = 0
for feature_name in feature_names:
    inputs = np.loadtxt(base_path + 'features/' + feature_name + '.txt')
    link = ward(inputs)
    tree = to_tree(link)

    # path to output
    out_path = base_path + 'clusters/' + feature_name + '/tree_cut/'

    for thresh in thresholds:
        lbls = np.ones(len(inputs), dtype='|S20')
        lbls = dist_cut_tree(tree, lbls, '1', all_feats, thresh, min_size=min_size, height_lim=height_lim)
        if n_clusters != len(np.unique(lbls)):
            n_clusters = len(np.unique(lbls))
            if not os.path.exists(base_path + 'clusters/' + feature_name + '/distcut/%d_clusters/' % n_clusters):
                os.mkdir(base_path + 'clusters/' + feature_name + '/distcut/%d_clusters/' % n_clusters)
            np.savetxt(base_path + 'clusters/' + feature_name + '/distcut/%d_clusters/clusters.txt' % n_clusters, lbls, fmt='%s')
    n_clusters = 0