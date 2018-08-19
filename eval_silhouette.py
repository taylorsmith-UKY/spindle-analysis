import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from sklearn.metrics import silhouette_score, silhouette_samples

# --------------------------------- PARAMETERS -----------------------------------#
base_path = 'cluster_data/'

paths = [('hyclass_pca', base_path + 'clusters/hyclass_pca/uniform/dyncut/104clusters_p1E-10/')]

nex = 10

# --------------------------------------------------------------------------------#
#
frequencies = np.loadtxt(base_path + 'features/frequencies.txt').reshape(-1, 1)
durations = np.loadtxt(base_path + 'features/durations.txt').reshape(-1, 1)
locations = np.loadtxt(base_path + 'features/locations.txt').reshape(-1, 1)
combined = np.hstack((frequencies, durations, locations))

fq_dm = squareform(pdist(frequencies, metric='cityblock'))
dur_dm = squareform(pdist(durations, metric='cityblock'))
loc_dm = squareform(pdist(locations, metric='cityblock'))
comp_dm = squareform(pdist(combined, metric='euclidean'))

for feature, path in paths:
    # Extract cluster features and construct distance matrix
    features = np.loadtxt(base_path + 'features/' + feature + '.txt')
    dm = pdist(features)
    sqdm = squareform(dm)
    cluster_labels = np.loadtxt(path + 'clusters.txt', dtype=str)
    label_names = np.unique(cluster_labels)

    fq_silh_glob = silhouette_samples(frequencies, cluster_labels, metric='cityblock')
    dur_silh_glob = silhouette_samples(durations, cluster_labels, metric='cityblock')
    loc_silh_glob = silhouette_samples(locations, cluster_labels, metric='cityblock')
    comp_silh_glob = silhouette_samples(combined, cluster_labels, metric='euclidean')
    
    out = open(path + 'silhouette_ref.csv', 'w')
    out.write('Parent_Name,Frequency_Pval,Duration_Pval,Location_Pval,mean_intra_left_feat,mean_intra_right_feat,mean_intra_left_fq,mean_intra_right_fq,mean_intra_left_dur,mean_intra_right_dur,mean_intra_left_loc,mean_intra_right_loc,mean_intra_left_comp,mean_intra_right_comp,left_min_inter_feat_loc,right_min_inter_feat_loc,left_min_inter_feat_glob,right_min_inter_feat_glob,left_min_inter_fq_loc,right_min_inter_fq_loc,left_min_inter_fq_glob,right_min_inter_fq_glob,left_min_inter_dur_loc,right_min_inter_dur_loc,left_min_inter_dur_glob,right_min_inter_dur_glob,left_min_inter_loc_loc,right_min_inter_loc_loc,left_min_inter_loc_glob,right_min_inter_loc_glob,left_min_inter_comp_loc,right_min_inter_comp_loc,left_min_inter_comp_glob,right_min_inter_comp_glob,left_feat_silh_loc,right_feat_silh_loc,left_feat_silh_glob,right_feat_silh_glob,left_fq_silh_loc,right_fq_silh_loc,left_fq_silh_glob,right_fq_silh_glob,left_dur_silh_loc,right_dur_silh_loc,left_dur_silh_glob,right_dur_silh_glob,left_loc_silh_loc,right_loc_silh_loc,left_loc_silh_glob,right_loc_silh_glob,left_comp_silh_loc,right_comp_silh_loc,left_comp_silh_glob,right_comp_silh_glob,left_frequency_silh_loc,right_frequency_silh_loc,left_duration_silh_loc,right_duration_silh_loc,left_location_silh_loc,right_location_silh_loc,left_combined_silh_loc,right_combined_silh_loc,left_frequency_silh_glob,right_frequency_silh_glob,left_duration_silh_glob,right_duration_silh_glob,left_location_silh_glob,right_location_silh_glob,left_combined_silh_glob,right_combined_silh_glob\n')
    print('Parent_Name,Frequency_Pval,Duration_Pval,Location_Pval,mean_intra_left_feat,mean_intra_right_feat,mean_intra_left_fq,mean_intra_right_fq,mean_intra_left_dur,mean_intra_right_dur,mean_intra_left_loc,mean_intra_right_loc,mean_intra_left_comp,mean_intra_right_comp,left_min_inter_feat_loc,right_min_inter_feat_loc,left_min_inter_feat_glob,right_min_inter_feat_glob,left_min_inter_fq_loc,right_min_inter_fq_loc,left_min_inter_fq_glob,right_min_inter_fq_glob,left_min_inter_dur_loc,right_min_inter_dur_loc,left_min_inter_dur_glob,right_min_inter_dur_glob,left_min_inter_loc_loc,right_min_inter_loc_loc,left_min_inter_loc_glob,right_min_inter_loc_glob,left_min_inter_comp_loc,right_min_inter_comp_loc,left_min_inter_comp_glob,right_min_inter_comp_glob,left_feat_silh_loc,right_feat_silh_loc,left_feat_silh_glob,right_feat_silh_glob,left_fq_silh_loc,right_fq_silh_loc,left_fq_silh_glob,right_fq_silh_glob,left_dur_silh_loc,right_dur_silh_loc,left_dur_silh_glob,right_dur_silh_glob,left_loc_silh_loc,right_loc_silh_loc,left_loc_silh_glob,right_loc_silh_glob,left_comp_silh_loc,right_comp_silh_loc,left_comp_silh_glob,right_comp_silh_glob,left_frequency_silh_loc,right_frequency_silh_loc,left_duration_silh_loc,right_duration_silh_loc,left_location_silh_loc,right_location_silh_loc,left_combined_silh_loc,right_combined_silh_loc,left_frequency_silh_glob,right_frequency_silh_glob,left_duration_silh_glob,right_duration_silh_glob,left_location_silh_glob,right_location_silh_glob,left_combined_silh_glob,right_combined_silh_glob')
    for tlbl in label_names:
        parent_name = '-'.join(tlbl.split('-')[:-1])
        left_name = tlbl
        right_name = tlbl
        if tlbl[-1] == '1':
            left_idx = np.where(cluster_labels == left_name)[0]

            right_name = right_name[:-1] + '2'
            if right_name in label_names:
                right_idx = np.where(cluster_labels == right_name)[0]
            else:
                all_right = np.array([right_name in x for x in label_names])
                sel = np.where(all_right)[0]
                right_idx = []
                for tsel in sel:
                    right_idx.append(np.where(cluster_labels == label_names[tsel])[0])
                right_idx = np.sort(np.concatenate(right_idx))

        else:
            right_idx = np.where(cluster_labels == right_name)[0]

            left_name = left_name[:-1] + '1'
            if left_name in label_names:
                left_idx = np.where(cluster_labels == left_name)[0]
            else:
                all_left = np.array([left_name in x for x in label_names])
                sel = np.where(all_left)[0]
                left_idx = []
                for tsel in sel:
                    left_idx.append(np.where(cluster_labels == label_names[tsel])[0])
                left_idx = np.sort(np.concatenate(left_idx))

        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        parent_idx = np.union1d(left_idx, right_idx)

        left_intra_idx = np.ix_(left_idx, left_idx)
        left_inter_loc_idx = np.ix_(left_idx, right_idx)
        left_inter_glob_idx = np.ix_(left_idx, np.setdiff1d(np.arange(len(features)), left_idx))
        
        left_intra_feat = sqdm[left_intra_idx]
        left_inter_loc_feat = sqdm[left_inter_loc_idx]
        left_inter_glob_feat = sqdm[left_inter_glob_idx]
        left_intra_fq = fq_dm[left_intra_idx]
        left_inter_loc_fq = fq_dm[left_inter_loc_idx]
        left_inter_glob_fq = fq_dm[left_inter_glob_idx]
        left_intra_dur = dur_dm[left_intra_idx]
        left_inter_loc_dur = dur_dm[left_inter_loc_idx]
        left_inter_glob_dur = dur_dm[left_inter_glob_idx]
        left_intra_loc = loc_dm[left_intra_idx]
        left_inter_loc_loc = loc_dm[left_inter_loc_idx]
        left_inter_glob_loc = loc_dm[left_inter_glob_idx]
        left_intra_comp = comp_dm[left_intra_idx]
        left_inter_loc_comp = comp_dm[left_inter_loc_idx]
        left_inter_glob_comp = comp_dm[left_inter_glob_idx]
        

        right_intra_idx = np.ix_(right_idx, right_idx)
        right_inter_loc_idx = np.ix_(right_idx, left_idx)
        right_inter_glob_idx = np.ix_(right_idx, np.setdiff1d(np.arange(len(features)), right_idx))

        right_intra_feat = sqdm[right_intra_idx]
        right_inter_loc_feat = sqdm[right_inter_loc_idx]
        right_inter_glob_feat = sqdm[right_inter_glob_idx]
        right_intra_fq = fq_dm[right_intra_idx]
        right_inter_loc_fq = fq_dm[right_inter_loc_idx]
        right_inter_glob_fq = fq_dm[right_inter_glob_idx]
        right_intra_dur = dur_dm[right_intra_idx]
        right_inter_loc_dur = dur_dm[right_inter_loc_idx]
        right_inter_glob_dur = dur_dm[right_inter_glob_idx]
        right_intra_loc = loc_dm[right_intra_idx]
        right_inter_loc_loc = loc_dm[right_inter_loc_idx]
        right_inter_glob_loc = loc_dm[right_inter_glob_idx]
        right_intra_comp = comp_dm[right_intra_idx]
        right_inter_loc_comp = comp_dm[right_inter_loc_idx]
        right_inter_glob_comp = comp_dm[right_inter_glob_idx]

        # aggregate minimum inter and mean intra-cluster distances
        left_intra_feat = np.mean(left_intra_feat, axis=1)
        left_inter_loc_feat = np.min(left_inter_loc_feat, axis=1)
        left_inter_glob_feat = np.min(left_inter_glob_feat, axis=1)
        left_intra_fq = np.mean(left_intra_fq, axis=1)
        left_inter_loc_fq = np.min(left_inter_loc_fq, axis=1)
        left_inter_glob_fq = np.min(left_inter_glob_fq, axis=1)
        left_intra_dur = np.mean(left_intra_dur, axis=1)
        left_inter_loc_dur = np.min(left_inter_loc_dur, axis=1)
        left_inter_glob_dur = np.min(left_inter_glob_dur, axis=1)
        left_intra_loc = np.mean(left_intra_loc, axis=1)
        left_inter_loc_loc = np.min(left_inter_loc_loc, axis=1)
        left_inter_glob_loc = np.min(left_inter_glob_loc, axis=1)
        left_intra_comp = np.mean(left_intra_comp, axis=1)
        left_inter_loc_comp = np.min(left_inter_loc_comp, axis=1)
        left_inter_glob_comp = np.min(left_inter_glob_comp, axis=1)

        right_intra_feat = np.mean(right_intra_feat, axis=1)
        right_inter_loc_feat = np.min(right_inter_loc_feat, axis=1)
        right_inter_glob_feat = np.min(right_inter_glob_feat, axis=1)
        right_intra_fq = np.mean(right_intra_fq, axis=1)
        right_inter_loc_fq = np.min(right_inter_loc_fq, axis=1)
        right_inter_glob_fq = np.min(right_inter_glob_fq, axis=1)
        right_intra_dur = np.mean(right_intra_dur, axis=1)
        right_inter_loc_dur = np.min(right_inter_loc_dur, axis=1)
        right_inter_glob_dur = np.min(right_inter_glob_dur, axis=1)
        right_intra_loc = np.mean(right_intra_loc, axis=1)
        right_inter_loc_loc = np.min(right_inter_loc_loc, axis=1)
        right_inter_glob_loc = np.min(right_inter_glob_loc, axis=1)
        right_intra_comp = np.mean(right_intra_comp, axis=1)
        right_inter_loc_comp = np.min(right_inter_loc_comp, axis=1)
        right_inter_glob_comp = np.min(right_inter_glob_comp, axis=1)

        left_intra_feat_agg = np.mean(left_intra_feat)
        left_inter_feat_agg_loc = np.mean(left_inter_loc_feat)
        left_inter_feat_agg_glob = np.mean(left_inter_glob_feat)
        left_intra_fq_agg = np.mean(left_intra_fq)
        left_inter_fq_agg_loc = np.mean(left_inter_loc_fq)
        left_inter_fq_agg_glob = np.mean(left_inter_glob_fq)
        left_intra_dur_agg = np.mean(left_intra_dur)
        left_inter_dur_agg_loc = np.mean(left_inter_loc_dur)
        left_inter_dur_agg_glob = np.mean(left_inter_glob_dur)
        left_intra_loc_agg = np.mean(left_intra_loc)
        left_inter_loc_agg_loc = np.mean(left_inter_loc_loc)
        left_inter_loc_agg_glob = np.mean(left_inter_glob_loc)
        left_intra_comp_agg = np.mean(left_intra_comp)
        left_inter_comp_agg_loc = np.mean(left_inter_loc_comp)
        left_inter_comp_agg_glob = np.mean(left_inter_glob_comp)
        
        right_intra_feat_agg = np.mean(right_intra_feat)
        right_inter_feat_agg_loc = np.mean(right_inter_loc_feat)
        right_inter_feat_agg_glob = np.mean(right_inter_glob_feat)
        right_intra_fq_agg = np.mean(right_intra_fq)
        right_inter_fq_agg_loc = np.mean(right_inter_loc_fq)
        right_inter_fq_agg_glob = np.mean(right_inter_glob_fq)
        right_intra_dur_agg = np.mean(right_intra_dur)
        right_inter_dur_agg_loc = np.mean(right_inter_loc_dur)
        right_inter_dur_agg_glob = np.mean(right_inter_glob_dur)
        right_intra_loc_agg = np.mean(right_intra_loc)
        right_inter_loc_agg_loc = np.mean(right_inter_loc_loc)
        right_inter_loc_agg_glob = np.mean(right_inter_glob_loc)
        right_intra_comp_agg = np.mean(right_intra_comp)
        right_inter_comp_agg_loc = np.mean(right_inter_loc_comp)
        right_inter_comp_agg_glob = np.mean(right_inter_glob_comp)
        

        left_frequencies = frequencies[left_idx]
        left_durations = durations[left_idx]
        left_locations = locations[left_idx]
        left_comp = combined[left_idx, :]

        right_frequencies = frequencies[right_idx]
        right_durations = durations[right_idx]
        right_locations = locations[right_idx]
        right_comp = combined[right_idx, :]

        _, freq_pval = ttest_ind(left_frequencies, right_frequencies)
        _, dur_pval = ttest_ind(left_durations, right_durations)
        _, loc_pval = ttest_ind(left_locations, right_locations)

        all_frequencies = np.concatenate((left_frequencies, right_frequencies)).reshape(-1, 1)
        all_durations = np.concatenate((left_durations, right_durations)).reshape(-1, 1)
        all_locations = np.concatenate((left_locations, right_locations)).reshape(-1, 1)
        all_comp = np.vstack((left_comp, right_comp))
        lbls = np.zeros(len(all_frequencies), dtype=int)
        lbls[len(left_idx):] = 1

        all_locref_freq_silh = silhouette_samples(all_frequencies, lbls, metric='cityblock')
        all_locref_dur_silh = silhouette_samples(all_durations, lbls, metric='cityblock')
        all_locref_loc_silh = silhouette_samples(all_locations, lbls, metric='cityblock')
        all_locref_comp_silh = silhouette_samples(all_comp, lbls, metric='euclidean')

        left_fq_silh_loc_ref = np.mean(all_locref_freq_silh[:len(left_idx)])
        left_dur_silh_loc_ref = np.mean(all_locref_dur_silh[:len(left_idx)])
        left_loc_silh_loc_ref = np.mean(all_locref_loc_silh[:len(left_idx)])
        left_comp_silh_loc_ref = np.mean(all_locref_comp_silh[:len(left_idx)])

        right_fq_silh_loc_ref = np.mean(all_locref_freq_silh[len(left_idx):])
        right_dur_silh_loc_ref = np.mean(all_locref_dur_silh[len(left_idx):])
        right_loc_silh_loc_ref = np.mean(all_locref_loc_silh[len(left_idx):])
        right_comp_silh_loc_ref = np.mean(all_locref_comp_silh[len(left_idx):])
        

        left_fq_silh_glob_ref = np.mean(fq_silh_glob[left_idx])
        left_dur_silh_glob_ref = np.mean(dur_silh_glob[left_idx])
        left_loc_silh_glob_ref = np.mean(loc_silh_glob[left_idx])
        left_comp_silh_glob_ref = np.mean(comp_silh_glob[left_idx])

        right_fq_silh_glob_ref = np.mean(fq_silh_glob[right_idx])
        right_dur_silh_glob_ref = np.mean(dur_silh_glob[right_idx])
        right_loc_silh_glob_ref = np.mean(loc_silh_glob[right_idx])
        right_comp_silh_glob_ref = np.mean(comp_silh_glob[right_idx])
        
        left_feat_silh_loc = (left_inter_feat_agg_loc - left_intra_feat_agg) / max(left_inter_feat_agg_loc,
                                                                                   left_intra_feat_agg)
        left_feat_silh_glob = (left_inter_feat_agg_glob - left_intra_feat_agg) / max(left_inter_feat_agg_glob,
                                                                                     left_intra_feat_agg)
        left_fq_silh_loc = (left_inter_fq_agg_loc - left_intra_fq_agg) / max(left_inter_fq_agg_loc,
                                                                             left_intra_fq_agg)
        left_fq_silh_glob = (left_inter_fq_agg_glob - left_intra_fq_agg) / max(left_inter_fq_agg_glob,
                                                                               left_intra_fq_agg)
        left_dur_silh_loc = (left_inter_dur_agg_loc - left_intra_dur_agg) / max(left_inter_dur_agg_loc,
                                                                                left_intra_dur_agg)
        left_dur_silh_glob = (left_inter_dur_agg_glob - left_intra_dur_agg) / max(left_inter_dur_agg_glob,
                                                                                  left_intra_dur_agg)
        left_loc_silh_loc = (left_inter_loc_agg_loc - left_intra_loc_agg) / max(left_inter_loc_agg_loc,
                                                                                left_intra_loc_agg)
        left_loc_silh_glob = (left_inter_loc_agg_glob - left_intra_loc_agg) / max(left_inter_loc_agg_glob,
                                                                                  left_intra_loc_agg)
        left_comp_silh_loc = (left_inter_comp_agg_loc - left_intra_comp_agg) / max(left_inter_comp_agg_loc,
                                                                                   left_intra_comp_agg)
        left_comp_silh_glob = (left_inter_comp_agg_glob - left_intra_comp_agg) / max(left_inter_comp_agg_glob,
                                                                                     left_intra_comp_agg)

        right_feat_silh_loc = (right_inter_feat_agg_loc - right_intra_feat_agg) / max(right_inter_feat_agg_loc,
                                                                                      right_intra_feat_agg)
        right_feat_silh_glob = (right_inter_feat_agg_glob - right_intra_feat_agg) / max(right_inter_feat_agg_glob,
                                                                                        right_intra_feat_agg)
        right_fq_silh_loc = (right_inter_fq_agg_loc - right_intra_fq_agg) / max(right_inter_fq_agg_loc,
                                                                                right_intra_fq_agg)
        right_fq_silh_glob = (right_inter_fq_agg_glob - right_intra_fq_agg) / max(right_inter_fq_agg_glob,
                                                                                  right_intra_fq_agg)
        right_dur_silh_loc = (right_inter_dur_agg_loc - right_intra_dur_agg) / max(right_inter_dur_agg_loc,
                                                                                   right_intra_dur_agg)
        right_dur_silh_glob = (right_inter_dur_agg_glob - right_intra_dur_agg) / max(right_inter_dur_agg_glob,
                                                                                     right_intra_dur_agg)
        right_loc_silh_loc = (right_inter_loc_agg_loc - right_intra_loc_agg) / max(right_inter_loc_agg_loc,
                                                                                   right_intra_loc_agg)
        right_loc_silh_glob = (right_inter_loc_agg_glob - right_intra_loc_agg) / max(right_inter_loc_agg_glob,
                                                                                            right_intra_loc_agg)
        right_comp_silh_loc = (right_inter_comp_agg_loc - right_intra_comp_agg) / max(right_inter_comp_agg_loc,
                                                                                   right_intra_comp_agg)
        right_comp_silh_glob = (right_inter_comp_agg_glob - right_intra_comp_agg) / max(right_inter_comp_agg_glob,
                                                                                     right_intra_comp_agg)

        print('%s,%.3E,%.3E,%.3E,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' %
              (parent_name, freq_pval, dur_pval, loc_pval,
               left_intra_feat_agg, right_intra_feat_agg, left_intra_fq_agg, right_intra_fq_agg,
               left_intra_dur_agg, right_intra_dur_agg, left_intra_loc_agg, right_intra_loc_agg, left_intra_comp_agg, right_intra_comp_agg,
               left_inter_feat_agg_loc, left_inter_feat_agg_glob, right_inter_feat_agg_loc,right_inter_feat_agg_glob,
               left_inter_fq_agg_loc, right_inter_fq_agg_loc, left_inter_fq_agg_glob, right_inter_fq_agg_glob,
               left_inter_dur_agg_loc, right_inter_dur_agg_loc, left_inter_dur_agg_glob, right_inter_dur_agg_glob,
               left_inter_loc_agg_loc, right_inter_loc_agg_loc, left_inter_loc_agg_glob, right_inter_loc_agg_glob,
               left_inter_comp_agg_loc, left_inter_comp_agg_glob, right_inter_comp_agg_loc, right_inter_comp_agg_glob,
               left_feat_silh_loc, right_feat_silh_loc, left_feat_silh_glob, right_feat_silh_glob,
               left_fq_silh_loc, right_fq_silh_loc, left_fq_silh_glob, right_fq_silh_glob,
               left_dur_silh_loc, right_dur_silh_loc, left_dur_silh_glob, right_dur_silh_glob,
               left_loc_silh_loc, right_loc_silh_loc, left_loc_silh_glob, right_loc_silh_glob,
               left_comp_silh_loc, right_comp_silh_loc, left_comp_silh_glob, right_comp_silh_glob,
               left_fq_silh_loc_ref, right_fq_silh_loc_ref, left_dur_silh_loc_ref, right_dur_silh_loc_ref,
               left_loc_silh_loc_ref, right_loc_silh_loc_ref, left_comp_silh_loc_ref, right_comp_silh_loc_ref,
               left_fq_silh_glob_ref, right_fq_silh_glob_ref, left_dur_silh_glob_ref, right_dur_silh_glob_ref,
               left_loc_silh_glob_ref, right_loc_silh_glob_ref, left_comp_silh_glob_ref, right_comp_silh_glob_ref))

        out.write('%s,%.3E,%.3E,%.3E,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' %
                    (parent_name, freq_pval, dur_pval, loc_pval,
                     left_intra_feat_agg, right_intra_feat_agg, left_intra_fq_agg, right_intra_fq_agg,
                     left_intra_dur_agg, right_intra_dur_agg, left_intra_loc_agg, right_intra_loc_agg, left_intra_comp_agg, right_intra_comp_agg,
                     left_inter_feat_agg_loc, left_inter_feat_agg_glob, right_inter_feat_agg_loc,right_inter_feat_agg_glob,
                     left_inter_fq_agg_loc, right_inter_fq_agg_loc, left_inter_fq_agg_glob, right_inter_fq_agg_glob,
                     left_inter_dur_agg_loc, right_inter_dur_agg_loc, left_inter_dur_agg_glob, right_inter_dur_agg_glob,
                     left_inter_loc_agg_loc, right_inter_loc_agg_loc, left_inter_loc_agg_glob, right_inter_loc_agg_glob,
                     left_inter_comp_agg_loc, left_inter_comp_agg_glob, right_inter_comp_agg_loc, right_inter_comp_agg_glob,
                     left_feat_silh_loc, right_feat_silh_loc, left_feat_silh_glob, right_feat_silh_glob,
                     left_fq_silh_loc, right_fq_silh_loc, left_fq_silh_glob, right_fq_silh_glob,
                     left_dur_silh_loc, right_dur_silh_loc, left_dur_silh_glob, right_dur_silh_glob,
                     left_loc_silh_loc, right_loc_silh_loc, left_loc_silh_glob, right_loc_silh_glob,
                     left_comp_silh_loc, right_comp_silh_loc, left_comp_silh_glob, right_comp_silh_glob,
                     left_fq_silh_loc_ref, right_fq_silh_loc_ref, left_dur_silh_loc_ref, right_dur_silh_loc_ref,
                     left_loc_silh_loc_ref, right_loc_silh_loc_ref, left_comp_silh_loc_ref, right_comp_silh_loc_ref,
                     left_fq_silh_glob_ref, right_fq_silh_glob_ref, left_dur_silh_glob_ref, right_dur_silh_glob_ref,
                     left_loc_silh_glob_ref, right_loc_silh_glob_ref, left_comp_silh_glob_ref, right_comp_silh_glob_ref))
