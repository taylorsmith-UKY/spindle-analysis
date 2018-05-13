from dynacut import *
import h5py
import sys
import os

sys.setrecursionlimit(10000)

f = h5py.File('data/spindles.h5','r')

if not os.path.exists('cluster_eval'):
    os.mkdir('cluster_eval')
if not os.path.exists('cluster_eval/encoded'):
    os.mkdir('cluster_eval/encoded')
if not os.path.exists('cluster_eval/bp_encoded'):
    os.mkdir('cluster_eval/bp_encoded')

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
totNbClusters = range(1, 26)
bad = 0
encoded = f['encoded_25'][:]
encoded_bp = f['csquare_vae_enc25'][:]

loss_dyn = np.zeros((len(totNbClusters), len(methods)))
loss_cut = np.zeros((len(totNbClusters), len(methods)))

bp_loss_dyn = np.zeros((len(totNbClusters), len(methods)))
bp_loss_cut = np.zeros((len(totNbClusters), len(methods)))

ct = 0
for nbClusters in totNbClusters:
    try:
        fname = 'cluster_eval/encoded/%d_clusters.txt' % nbClusters
        (loss_dyn[ct, :], loss_cut[ct, :]) = bench_methods(encoded, nbClusters, methods, fname)
    except:
        bad += 1
        loss_dyn[ct, :] = np.nan
        loss_cut[ct, :] = np.nan
    try:
        fname = 'cluster_eval/bp_encoded/%d_clusters.txt' % nbClusters
        (bp_loss_dyn[ct, :], bp_loss_cut[ct, :]) = bench_methods(encoded_bp, nbClusters, methods, fname)
    except:
        bad += 1
        bp_loss_dyn[ct, :] = np.nan
        bp_loss_cut[ct, :] = np.nan
    ct += 1
print('%d\n\n' % bad)

# Print the summary results
print('Dynamic Loss - No Bandpass')
print('nbClusters,single,complete,average,weighted,centroid,median,ward')
for i in range(len(totNbClusters)):
    print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, loss_dyn[i, 0], loss_dyn[i,1], loss_dyn[i, 2], loss_dyn[i, 3],
                                                     loss_dyn[i, 4], loss_dyn[i, 5], loss_dyn[i, 6]))
print('\n')
print('Dynamic Loss - With Bandpass')
print('nbClusters,single,complete,average,weighted,centroid,median,ward')
for i in range(len(totNbClusters)):
    print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, bp_loss_dyn[i, 0], bp_loss_dyn[i,1], bp_loss_dyn[i, 2],
                                                     bp_loss_dyn[i, 3], bp_loss_dyn[i, 4], bp_loss_dyn[i, 5],
                                                     bp_loss_dyn[i, 6]))

print('Straight Cut - No Bandpass')
print('nbClusters,single,complete,average,weighted,centroid,median,ward')
for i in range(len(totNbClusters)):
    print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, loss_cut[i, 0], loss_cut[i,1], loss_cut[i, 2], loss_cut[i, 3],
                                                     loss_cut[i, 4], loss_cut[i, 5], loss_cut[i, 6]))
print('\n')
print('Straight Cut - With Bandpass')
print('nbClusters,single,complete,average,weighted,centroid,median,ward')
for i in range(len(totNbClusters)):
    print('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (i+1, bp_loss_cut[i, 0], bp_loss_cut[i,1], bp_loss_cut[i, 2],
                                                     bp_loss_cut[i, 3], bp_loss_cut[i, 4], bp_loss_cut[i, 5],
                                                     bp_loss_cut[i, 6]))
