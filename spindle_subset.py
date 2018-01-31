#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:50:53 2018

@author: taylorsmith
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

dataFile = 'clustering/test/spindles.h5'
testFile = 'clustering/test/test_setA.h5'

n_per_grp = 200
sel_grp = 'freq_amp_max'
boundaries = ((8,12),(12,16))
ch_sel = 8      #8 is channel C3



f=h5py.File(dataFile,'r')
stats=f['spindle_stats']
sel_stat = stats[sel_grp]
if len(sel_stat.shape) > 1:
    sel_stat = sel_stat[:,ch_sel]


sel_stat_sorted = np.sort(sel_stat)
sorted_order = np.argsort(sel_stat)

grps = []
for i in range(len(boundaries)):
    grps.append(np.intersect1d(np.where(sel_stat_sorted > boundaries[i][0]),\
                               np.where(sel_stat_sorted < boundaries[i][1])))

ids = []
for i in range(len(grps)):
    try:
        r = random.sample(range(len(grps[i])),n_per_grp)
    except:
        r = range(len(grps[i]))
    rando=grps[i][r]
    ids.append(sorted_order[rando])

all_ids = list(np.unique(np.concatenate(ids)))
sel = np.zeros(len(sel_stat),dtype=bool)
for idx in all_ids:
    sel[idx] = 1

out = h5py.File(testFile,'w')
keys = f.keys()
for k in keys:
    ds = f[k]
    try:
        sub_keys = ds.keys()
        grp = out.create_group(ds.name)
        for sk in sub_keys:
            grp.create_dataset(str(sk),data=ds[sk][all_ids])
    except:
        out.create_dataset(str(k),data=ds[all_ids])

dur = out['spindle_stats']['durations'][:]
ton = out['spindle_stats']['time_of_night'][:]/256
fq_dist = out['spindle_stats']['freq_amp_max'][:,ch_sel]



fig = plt.figure(figsize=(7,3))
gs = gridspec.GridSpec(1, 3)
plt.subplot(gs[0])
plt.hist(dur,bins=n_per_grp/12)
plt.title('Duration')
plt.xlabel('Length (s)')
plt.ylabel('# of Spindles')
plt.subplot(gs[1])
plt.hist(fq_dist,bins=n_per_grp/12)
plt.title('Dominant Spindle Freq.')
plt.xlabel('Freq (Hz)')
plt.subplot(gs[2])
plt.hist(ton,bins=n_per_grp/12)
plt.title('Time of Night',y=1.02)
plt.xlabel('Time Passed (Hr)')
plt.suptitle('Overview of test set with '+str(len(all_ids))+' different spindles')
plt.tight_layout(pad=0.5,rect=(0,0,1,0.95))

figFile = testFile.split('.')[0] + '.png'
plt.savefig(figFile)
#plt.savefig('vae_reconstruction_ex'+str(i+1)+'.png')


print testFile

