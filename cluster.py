#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:50:43 2018

@author: taylorsmith
"""

import cluster_funcs as cf
import json
import numpy as np
import h5py
import os

n_clusts=5
new_groups = True


f=open('conf.json','r')
conf=json.load(f)
f.close()

output_path=conf['output_path']
f_crop=conf['f_crop']
latent_dim=conf['latent_dim']

data_grp='encoded_'+str(latent_dim)

cdata_path = output_path + data_grp + '/'

if not os.path.exists(cdata_path):
    os.makedirs(cdata_path)

df = h5py.File(output_path+'test_set.h5','r')

ds = df[data_grp]

n_ex = ds.shape[0]
ids = np.arange(n_ex)

ids = np.loadtxt('test_ids.txt',dtype=int)

if new_groups:
    link = np.loadtxt(cdata_path+'linkage_data.txt')
else:
    link = cf.cluster(ds,ids,cdata_path)
    cf.plot_dend(link,cdata_path)

fig_path=cdata_path+str(n_clusts)+'_clusters/'
if not os.path.exists(cdata_path):
    os.makedirs(cdata_path)


cf.clust_grps(n_clusts,cdata_path)

stats = df['spindle_stats']
cf.cluster_stats(stats,fig_path)

