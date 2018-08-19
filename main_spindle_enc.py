import preprocess_funcs as ppf
import spindle_ae as sae
import h5py
import numpy as np
import os
import json
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint

# ------------------------------- PARAMETERS -----------------------------------#
# Read configuration defining input directory and filenames, etc.
f = open('conf.json', 'r')
conf = json.load(f)
f.close()

data_path = conf['data_path']  # input path to edf files
output_path = conf['output_path']  # top directory for processed data
sigfiles = conf['signal_files']  # list of source signal files
detfiles = conf['spindle_annotations']  # list of spindle annotations
bp_fstop1 = conf['bp_fstop1']
bp_fstop2 = conf['bp_fstop2']
nfft = conf['nfft']
ham_win = conf['ham_win']
f_crop = conf['f_crop']
scale_factors = conf['scale_factors']
netFilename = conf['netFilename']
gpu_id = conf['gpu_id']
TB = conf['TB']
vae_loc = conf['vae_loc']
n_ex = conf['n_ex']
batch_size = conf['batch_size']
val_split = conf['val_split']
n_epochs = conf['n_epochs']
n_eval = conf['n_eval']
t_flag = conf['train']
latent_dim = conf['latent_dim']
epsln_std = conf['epsilon_std']

for df in detfiles:
    for i in range(len(df)):
        df[i] = data_path + df[i]
for i in range(len(sigfiles)):
    sigfiles[i] = data_path + sigfiles[i]

dataFile = output_path + 'spindles_bp.h5'
spec_name = 'spectrograms_bp_' + str(f_crop[0]) + '-' + str(f_crop[1])
ds_name = 'spectrograms_bp_' + str(f_crop[0]) + '-' + str(f_crop[1]) + '_norm_sqr_concat_rsz'
vae_name = vae_loc + netFilename
# ------------------------------------------------------------------------------#

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(vae_loc):
    os.makedirs(vae_loc)
try:
    f = h5py.File(dataFile, 'r+')
    spindles = f['spindles']
except:
    spindles = ppf.get_spindles(sigfiles, detfiles, dataFile)
try:
    ds = f[ds_name]
    # specs = f[spec_name]
except:
    try:
        concat = f[spec_name + '_norm_sqr_concat']
    except:
        try:
            sqr = f[spec_name + '_norm_sqr']
        except:
            try:
                norm = f[spec_name + '_norm']
            except:
                try:
                    specs = f[spec_name]
                except:
                    try:
                        bp = f['spindles_bp']
                    except:
                        bp = ppf.bandpass_signals(spindles, bp_fstop1, bp_fstop2)
                    specs = ppf.fft_fe(bp, nfft, ham_win, f_crop, spec_name)
                norm = ppf.normalize(specs)
            sqr = ppf.square_pad(norm)
        concat = ppf.concat_square_specs(sqr)
    ds = ppf.resize_specs(concat, scale_factors=(2, 2), channels_last=True)

# ppf.spindle_stats(specs, f_range=f_crop)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
vae, vae_loss = sae.get_conv_vae(ds.shape[1:])  # returns a compiled Keras model, along with custom loss function
rds = HDF5Matrix(dataFile, ds_name)
model_checkpoint = ModelCheckpoint(vae_name, monitor='loss', verbose=1, save_best_only=True)
history = vae.fit(rds, rds, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.2,
                  shuffle='batch', callbacks=[model_checkpoint])

loss = history.history['loss']
loss = np.array(loss)
lossFilename = vae_name.split('.')[0] + '_loss_log.txt'
np.savetxt(lossFilename, loss, fmt='%.4f')

sae.save_ex(vae, ds, n_eval, vae_loc)
encoded = sae.encode_data(vae, ds)
f.create_dataset('encoded_' + str(latent_dim), data=encoded)

ds_name_conv = 'spectrograms_' + str(f_crop[0]) + '-' + str(f_crop[1]) + '_rsz_norm_concat'
vae_name_conv = vae_name.split('.')[0] + 'conv.h5'
ds = f[ds_name_conv]
vae = sae.train_vae(ds, dataFile, ds_name_conv, vae_name_conv, n_epochs=n_epochs, val_split=val_split, conv=True)
sae.save_ex(vae, ds, n_eval, vae_loc)
encoded = sae.encode_data(vae, ds)
f.create_dataset('conv_encoded_' + str(latent_dim), data=encoded)
