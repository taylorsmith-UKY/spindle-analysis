#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:42:12 2018

@author: taylorsmith
"""

from spindle_ae import get_vae, get_conv_vae
from keras.callbacks import ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
from keras.models import load_model
import numpy as np
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
vae_type=conf['vae_type']
orig_dim=conf['original_dim']
batch_size=conf['batch_size']
spindleFilename=conf['spindleFilename']

dataFile = output_path + spindleFilename
vae_name = vae_loc + netFilename

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

if vae_type[:4].lower() == 'trad':
    ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat_flat'
    l=np.prod(orig_dim)
    vae,vae_loss = get_vae(l)
elif vae_type[:4].lower() == 'conv':
    ds_name = 'spectrograms_bp_8-17_norm_square_concat'
    vae,vae_loss = get_conv_vae(orig_dim)

rds = HDF5Matrix(dataFile,ds_name)

if os.path.exists(vae_name):
    if vae_type[:4].lower() == 'trad':
        vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})
    elif vae_type[:4].lower() == 'conv':
        vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'cvae_loss': vae_loss})
    vae_name = vae_name.split('.')[0]+'.h5'

model_checkpoint = ModelCheckpoint(vae_name, monitor='loss',verbose=1, save_best_only=True)

history = vae.fit(rds, rds, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.2, shuffle='batch', callbacks=[model_checkpoint])
loss = history.history['loss']
loss = np.array(loss)
lossFilename = vae_name.split('.')[0] + '_loss_log.txt'
np.savetxt(lossFilename,loss,fmt='%.4f')
