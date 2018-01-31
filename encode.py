#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:02:53 2018

@author: taylorsmith
"""
import h5py
from spindle_ae import encode_data, get_vae
from keras.models import load_model
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

ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat_flat'
dataFile = output_path + 'test_set.h5'
vae_name = vae_loc + netFilename

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

f = h5py.File(dataFile,'r+')
ds = f[ds_name]

npts = len(ds[0,:])

_, vae_loss = get_vae(npts)

vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})

encoded = encode_data(vae,ds)

enc_ds = f.create_dataset('encoded_'+str(latent_dim),data=encoded)

f.close()