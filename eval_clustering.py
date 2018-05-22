from dynacut import *
import h5py
import sys
import os

sys.setrecursionlimit(10000)

#######################
data_file = 'data/spindles.h5'
ds_names = ['encoded_25bp', 'c3_freq_dist_bp', 'encoded_25', 'c3_freq_dist_new']

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
totNbClusters = range(2, 25)
#######################


f = h5py.File(data_file,'r')

if not os.path.exists('cluster_eval'):
    os.mkdir('cluster_eval')


for ds in ds_names:
    bad = 0
    data = f[ds][:]
    tpath = 'cluster_eval/' + ds + '/'

    loss_dyn = np.zeros((len(totNbClusters), len(methods)))
    loss_cut = np.zeros((len(totNbClusters), len(methods)))

    if not os.path.exists('cluster_eval/encoded'):
        os.mkdir('cluster_eval/encoded')

    ct = 0
    for nbClusters in totNbClusters:
        try:
            fname = tpath + '%d_clusters.txt' % nbClusters
            (loss_dyn[ct, :], loss_cut[ct, :]) = bench_methods(data, nbClusters, methods, fname)
        except:
            bad += 1
            loss_dyn[ct, :] = np.nan
            loss_cut[ct, :] = np.nan
        ct += 1
    print('%d\n\n' % bad)

    log = open(tpath + 'dyn_loss.csv', 'w')
    # Print the summary results
    print('Dynamic Loss')
    print('nbClusters,single,complete,average,weighted,centroid,median,ward')
    log.write('nbClusters,single,complete,average,weighted,centroid,median,ward\n')
    for i in range(len(totNbClusters)):
        print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, loss_dyn[i, 0], loss_dyn[i,1], loss_dyn[i, 2],
                                                         loss_dyn[i, 3], loss_dyn[i, 4], loss_dyn[i, 5],
                                                         loss_dyn[i, 6]))
        log.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (i + 1, loss_dyn[i, 0], loss_dyn[i, 1], loss_dyn[i, 2],
                                                               loss_dyn[i, 3], loss_dyn[i, 4], loss_dyn[i, 5],
                                                               loss_dyn[i, 6]))
    log.close()

    log = open(tpath + 'cut_loss.csv', 'w')
    print('Straight Cut')
    print('nbClusters,single,complete,average,weighted,centroid,median,ward')
    log.write('nbClusters,single,complete,average,weighted,centroid,median,ward\n')
    for i in range(len(totNbClusters)):
        print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, loss_cut[i, 0], loss_cut[i,1], loss_cut[i, 2],
                                                         loss_cut[i, 3], loss_cut[i, 4], loss_cut[i, 5],
                                                         loss_cut[i, 6]))
        log.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (i + 1, loss_cut[i, 0], loss_cut[i, 1], loss_cut[i, 2],
                                                               loss_cut[i, 3], loss_cut[i, 4], loss_cut[i, 5],
                                                               loss_cut[i, 6]))
    log.close()
