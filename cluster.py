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

n_clusts=6
new_groups = True


f=open('conf.json','r')
conf=json.load(f)
f.close()

output_path=conf['output_path']
f_crop=conf['f_crop']
latent_dim=conf['latent_dim']
vae_type=conf['vae_type']
spindleFilename=conf['spindleFilename']

data_grp='vae_enc'+str(latent_dim)
if vae_type[:4].lower() == 'conv':
    data_grp = 'c'+data_grp

cdata_path = 'data/square_cvae_enc25/'

if not os.path.exists(cdata_path):
    os.makedirs(cdata_path)

df = h5py.File(output_path+spindleFilename,'r')

ds = df['encoded_25']

n_ex = ds.shape[0]
ids = np.arange(n_ex)

if os.path.exists(cdata_path+'linkage_data.txt'):
    link = np.loadtxt(cdata_path+'linkage_data.txt')
else:
    link = cf.cluster(ds,ids,cdata_path)
    cf.plot_dend(link,cdata_path)

fig_path=cdata_path+str(n_clusts)+'_clusters/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

cf.clust_grps(link, n_clusts, fig_path)

stats = df['spindle_stats']
cf.cluster_stats(stats,fig_path)
