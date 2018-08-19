#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 11:51:51 2018

@author: taylorsmith
"""

import h5py
from spindle_ae import reconstruct, get_vae, get_conv_vae, kl_divergence, mutual_info
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
import numpy as np
import json
import os

f = open('conf.json', 'r')
conf = json.load(f)
f.close()

# ------------------------------------ PARAMETERS ------------------------------------- #
output_path = 'data/bandpass/cvae/'
f_crop = (8, 16)
latent_dim = 25
epsln_std = 1.0
vae_name = 'data/bandpass/cvae/vae_model_conv.h5'
gpu_id = '4,5,6,7'
vae_type = 'conv'
orig_dim = (113, 113, 3)
batch_size = 32
dataFile = 'data/spindles.h5'
ds_name = 'spectrograms_bp_8-17_norm_square_concat'

# ------------------------------------------------------------------------------------- #


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

enc_name = 'vae_enc' + str(latent_dim)
if vae_type[:4].lower() == 'trad':
    l = np.prod(orig_dim)
    _, vae_loss = get_vae(l)
    vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})
elif vae_type[:4].lower() == 'conv':
    _, vae_loss = get_conv_vae(orig_dim)
    vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'cvae_loss': vae_loss})

f = h5py.File(dataFile, 'r+')

ds = f[ds_name][:]

recon = reconstruct(vae, ds)

mi = np.zeros(len(recon))
kl = np.zeros(len(recon))
for i in range(len(recon)):
    mi[i] = mutual_info(ds[i, :, :, :], recon[i, :, :, :])
    kl[i] = kl_divergence(ds[i, :, :, :], recon[i, :, :, :])
np.savetxt(output_path + 'mutual_info_scores.txt', mi, fmt='%.4f')
np.savetxt(output_path + 'kl_divergence_scores.txt', kl, fmt='%.4f')

print('Mutual Info\tMean: %.3f\tStd: %.3f' % (float(np.mean(mi)), float(np.std(mi))))
print('K-L Divergence\tMean: %.3f\tStd: %.3f' % (float(np.mean(mi)), float(np.std(mi))))

f.close()
