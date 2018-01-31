import preprocess_funcs as ppf
import spindle_ae as sae
import h5py
#import numpy as np
import os
import json
#------------------------------- PARAMETERS -----------------------------------#
#Read configuration defining input directory and filenames, etc.
f=open('conf.json','r')
conf=json.load(f)
f.close()

data_path=conf['data_path']                #input path to edf files
output_path=conf['output_path']             #top directory for processed data
sigfiles=conf['signal_files']               #list of source signal files
detfiles=conf['spindle_annotations']        #list of spindle annotations
nfft=conf['nfft']
ham_win=conf['ham_win']
f_crop=conf['f_crop']
scale_factors=conf['scale_factors']
netFilename=conf['netFilename']
gpu_id=conf['gpu_id']
TB=conf['TB']
vae_loc=conf['vae_loc']
n_ex=conf['n_ex']
batch_size=conf['batch_size']
val_split=conf['val_split']
n_epochs=conf['n_epochs']
n_eval=conf['n_eval']
t_flag=conf['train']
latent_dim=conf['latent_dim']
epsln_std=conf['epsilon_std']

for df in detfiles:
	for i in range(len(df)):
		df[i]=data_path+df[i]
for i in range(len(sigfiles)):
	sigfiles[i]=data_path+sigfiles[i]

dataFile = output_path + 'spindles.h5'
spec_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])
#spec_name = 'inputs'
ds_name = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat_flat'
vae_name = vae_loc + netFilename
#------------------------------------------------------------------------------#

def main():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(vae_loc):
        os.makedirs(vae_loc)
    try:
        f = h5py.File(dataFile,'r+')
        spindles=f['spindles']
    except:
        spindles=ppf.get_spindles(sigfiles,detfiles,dataFile)
    try:
        flat=f[spec_name+'_rsz_norm_concat_flat']
    except:
        try:
            concat=f[spec_name+'_rsz_norm_concat']
        except:
            try:
                norm=f[spec_name+'_rsz_norm']
            except:
                try:
                    rsz = f[spec_name+'_rsz']
                except:
                    try:
                        specs=f[spec_name]
                    except:
                        specs=ppf.fft_fe(spindles,nfft,ham_win,f_crop,spec_name)
                    rsz = ppf.resize_specs(specs,scale_factors)
                norm=ppf.normalize(rsz)
            concat = ppf.concat_specs(norm)
        ppf.flatten_feats(concat)
    specs = f[spec_name]
    ppf.spindle_stats(specs)
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    ds = f[ds_name]
    vae = sae.train_vae(ds,dataFile,ds_name,vae_name,n_epochs=n_epochs,val_split=val_split)
    sae.save_ex(vae,ds,n_eval,vae_loc)
    encoded = sae.encode_data(vae,ds)
    f.create_dataset('encoded_'+str(latent_dim),data=encoded)

    ds_name_conv = 'spectrograms_'+str(f_crop[0])+'-'+str(f_crop[1])+'_rsz_norm_concat'
    vae_name_conv = vae_name.split('.')[0]+'conv.h5'
    ds = f[ds_name_conv]
    vae = sae.train_vae(ds,dataFile,ds_name_conv,vae_name_conv,n_epochs=n_epochs,val_split=val_split,conv=True)
    sae.save_ex(vae,ds,n_eval,vae_loc)
    encoded = sae.encode_data(vae,ds)
    f.create_dataset('conv_encoded_'+str(latent_dim),data=encoded)
    return


main()