#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:02:53 2018

@author: taylorsmith
"""
import h5py
from spindle_ae import encode_data, get_vae, get_conv_vae
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
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
vae_type=conf['vae_type']
orig_dim=conf['original_dim']
batch_size=conf['batch_size']
spindleFilename=conf['spindleFilename']

dataFile = output_path + spindleFilename
vae_name = vae_loc + netFilename

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id


enc_name = 'vae_enc'+str(latent_dim)
if vae_type[:4].lower() == 'trad':
    ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat_flat'
    l=np.prod(orig_dim)
    _,vae_loss = get_vae(l)
    vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})
elif vae_type[:4].lower() == 'conv':
    enc_name = 'c'+enc_name
    ds_name = 'spectrograms_bp_8-17_norm_square_concat'
    _,vae_loss = get_conv_vae(orig_dim)
    vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'cvae_loss': vae_loss})

ds = HDF5Matrix(dataFile,ds_name)

encoded = encode_data(vae,ds)

f = h5py.File(dataFile,'r+')

enc_ds = f.create_dataset(enc_name,data=encoded)

f.close()