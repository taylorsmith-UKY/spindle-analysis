#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:42:12 2018

@author: taylorsmith
"""

from spindle_ae import get_vae, get_conv_vae
from keras.callbacks import ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
import h5py
import json
import os

f=open('conf.json','r')
conf=json.load(f)
f.close()

output_path=conf['output_path']
f_crop=conf['f_crop']
latent_dim=conf['latent_dim']
epsln_std=conf['epsilon_std']
vae_loc=conf['vae_loc']
netFilename=conf['netFilename']
gpu_id=conf['gpu_id']
n_epochs=conf['n_epochs']

ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat_flat'
#ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat'
dataFile = output_path + 'spindles.h5'
vae_name = vae_loc + netFilename

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

f=h5py.File(dataFile,'r')
rds=f[ds_name]
l = rds.shape[1]
vae,vae_loss = get_vae(l)
#vae,vae_loss = get_vae((299,299,3))
f.close()

rds = HDF5Matrix(dataFile,ds_name)

model_checkpoint = ModelCheckpoint(vae_name, monitor='loss',verbose=1, save_best_only=True)

vae.fit(rds, rds, epochs=n_epochs, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint]))