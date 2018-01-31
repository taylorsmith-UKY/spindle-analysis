#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:28:05 2018

@author: taylorsmith
"""
import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Lambda,\
                            Dropout, UpSampling2D, Concatenate,Conv2DTranspose,\
                            Flatten, Reshape
from keras import optimizers as opt
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.utils.io_utils import HDF5Matrix
import random
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import h5py

f=open('conf.json','r')
conf=json.load(f)
f.close()

#------------------------------- PARAMETERS -----------------------------------#
#Convolutional VAE
filters = conf['n_filters']
n_conv = conf['n_conv']

#Variational AE
original_dim = np.product(conf['original_dim'])
intermediate_dim = conf['intermediate_dim']
latent_dim = conf['latent_dim']
init_lr = conf['init_lr']
epsln_std = conf['epsilon_std']
batch_size = conf['batch_size']
#------------------------------------------------------------------------------#

#%%         Standard Variational AE
# Custom loss layer
def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsln_std)
        return z_mean + K.exp(z_log_var) * epsilon

def get_vae(in_shape):
    inputs = Input((in_shape,),name='prim_input')
    h = Dense(intermediate_dim, activation='relu',name='intmdt_1')(inputs)
    z_mean = Dense(latent_dim,name='z_mean')(h)
    z_log_var = Dense(latent_dim,name='z_std')(h)

    z = Lambda(sampling,name='sampling')([z_mean,z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu',name='decoder_1')
    decoder_mean = Dense(in_shape, activation='sigmoid',name='decoder_2/out')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(inputs,x_decoded_mean)


    def vae_loss(x, x_decoded_mean):

        xent_loss = np.prod(original_dim) * metrics.mse(K.flatten(x), K.flatten(x_decoded_mean))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    vae.compile(optimizer=opt.Adam(lr=init_lr), loss=vae_loss)

    return vae, vae_loss



#%% Convolutional AE
def get_conv_vae(in_shape):
    img_rows, img_cols, img_chns = in_shape
    x = Input(in_shape)
    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
#    conv_3 = Conv2D(filters,
#                    kernel_size=n_conv,
#                    padding='same', activation='relu',
#                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=n_conv,
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_2)
    enc_dim = int(conv_4.shape[1])

    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim,name='z_mean')(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling)([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * enc_dim * enc_dim, activation='relu')

    output_shape = (batch_size, enc_dim, enc_dim, filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1_upsamp = Conv2DTranspose(filters,
                                       kernel_size=n_conv,
                                       padding='same',
                                       strides=(2,2),
                                       activation='relu')
    #decoder_deconv_2 = Conv2DTranspose(filters,
    #                                   kernel_size=n_conv,
    #                                   padding='same',
    #                                   strides=1,
    #                                   activation='relu')

    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=(3,3),
                                 padding='valid',
                                 activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1_upsamp(reshape_decoded)
    #deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_1_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    # instantiate VAE model
    vae = Model(x, x_decoded_mean_squash)

    # Compute VAE loss
    def cvae_loss(x, x_decoded_mean_squash):
        xent_loss = np.prod(original_dim) * metrics.mse(
            K.flatten(x),
            K.flatten(x_decoded_mean_squash))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    #vae.add_loss(cvae_loss)

    vae.compile(optimizer=opt.Adam(lr=init_lr), loss=cvae_loss)

    return vae, cvae_loss



#%%
def save_ex(vae_name,ds,n_eval,log_loc,tar_name='vae_eval.tar.gz'):
    tot_ex = ds.shape[0]
    npts = ds.shape[1]

    _,  vae_loss = get_vae(npts)

    if ds.shape > 2:
        vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'cvae_loss': vae_loss})
    else:
        vae = load_model(vae_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})

    test = []
    test = np.sort(random.sample(range(tot_ex),n_eval))

    test_img = []
    for i in range(n_eval):
        test_img.append(ds[test[i]])
    test_img = np.array(test_img)
    test_rcnstrct = vae.predict(test_img)
    enc = Model(inputs=vae.input,outputs=vae.get_layer('dense_2').output)
    test_enc = enc.predict(test_img)

    o_log = open(log_loc+'orig.csv','w')
    re_log = open(log_loc+'recon.csv','w')
    enc_log = open(log_loc+'latent.csv','w')

    for i in range(n_eval):
        o_log.write('Example #'+str(test[i])+'\n')
        re_log.write('Example #'+str(test[i])+'\n')
        enc_log.write('Example #'+str(test[i])+'\n')
        if ds.shape == 2:
            orig = np.reshape(test_img[i,:],(299,299,3))
        else:
            orig = test_img[i,:,:,:]
        print_mat(o_log,orig[:,:,0])
        o_log.write('\n')
        print_mat(o_log,orig[:,:,1])
        o_log.write('\n')
        print_mat(o_log,orig[:,:,2])
        o_log.write('\n')

        if ds.shape == 2:
            recon = np.reshape(test_rcnstrct[i,:],(299,299,3))
        else:
            recon = test_rcnstrct[i,:,:,:]
        print_mat(re_log,recon[:,:,0])
        re_log.write('\n')
        print_mat(re_log,recon[:,:,1])
        re_log.write('\n')
        print_mat(re_log,recon[:,:,2])
        re_log.write('\n')

        print_mat(enc_log,test_enc[i])

    o_log.close()
    re_log.close()
    enc_log.close()

    call(['tar','-zcvf',log_loc+tar_name,log_loc+'orig.csv',log_loc+'recon.csv',log_loc+'latent.csv'])
    return


def print_mat(f,mat,sep=','):
    shp = np.shape(mat)
    if len(shp) == 2:
        for i in range(shp[0]):
            f.write('%f' % (mat[i,0]))
            for j in range(1,shp[1]):
                f.write('%s%f' % (sep,mat[i,j]))
            f.write('\n')
    elif len(shp) == 1:
        f.write('%f' % (mat[0]))
        for i in range(1,len(mat)):
            f.write('%s%f' % (sep,mat[i]))
        f.write('\n')
    return

def vae_vis(data_path):
    x = read_csv(data_path+'orig.csv',3,299)
    x_p = read_csv(data_path+'recon.csv',3,299)
    f=open(data_path+'latent.csv')
    enc = []
    for l in f:
        if len(l.split(',')) > 1:
            enc.append(np.array(l.rstrip().split(','),dtype=float))
    #enc = read_csv(data_path+'latent.csv')
    n_eval = x.shape[0]
    for i in range(n_eval):
        fig = plt.figure(figsize=(3,8))
        gs = gridspec.GridSpec(5, 1)
        fig.subplots_adjust(hspace=.4)
        ax1 = plt.subplot(gs[:2])
        plt.pcolormesh(x[i,:,:,1],vmin=0,vmax=1)
        plt.title('Original')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = plt.subplot(gs[2:4])
        plt.pcolormesh(x_p[i,:,:,1],vmin=0,vmax=1)
        plt.title('Reconstructed')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3 = plt.subplot(gs[4])
        plt.pcolormesh([enc[i],enc[i]])
        plt.title('Latent Representation')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.suptitle('Example #'+str(i+1))
        plt.savefig(data_path+'vae_reconstruction_ex'+str(i+1)+'.png')


def read_csv(fname, dims, height):
    f=open(fname,'r')
    ds = []     #4-D array
    mat = []    #2-D array
    dmat = []   #3-D array
    arr = []    #1-D array
    count = 0
    for l in f:
        l = l.rstrip()
        if len(l.split()) == 2: #new window
            if ds == [] and dmat != []:  #very first window completed
                dmat = np.expand_dims(dmat,0)
                ds = dmat
            elif ds != []:              #add 3-D image to dataset
                dmat = np.expand_dims(dmat,0)
                ds = np.vstack((ds,dmat))
            mat = []
            dmat = []
        elif len(l.split(',')) > 1:
            arr = np.array(l.split(','),dtype=float)
            count += 1
            if mat == []:
                mat = arr
            else:
                mat = np.vstack((mat,arr))
                if count == height:
                    if dmat == []:
                        dmat = mat
                    else:
                        dmat = np.dstack((dmat,mat))
                    mat = []
                    count = 0
    dmat = np.expand_dims(dmat,0)
    ds = np.vstack((ds,dmat))
    return ds

#%%
def train_vae(rds,dataFile,ds_name,vae_name,n_epochs=10,val_split=0.2,conv=False):
    if conv:
        shp = rds.shape[1:]
        vae,vae_loss = get_conv_vae(shp)
    else:
        l = rds.shape[1]
        vae,vae_loss = get_vae(l)
    f.close()

    ds = HDF5Matrix(dataFile,ds_name)
    model_checkpoint = ModelCheckpoint(vae_name, monitor='loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    vae.fit(ds, ds, epochs=n_epochs, verbose=1, validation_split=val_split, shuffle='batch', callbacks=[model_checkpoint])
    print('Model saved to \''+vae_name+'\'')
    return vae

def cont_training(load_name,save_name,in_shape,hdf_name,ds_name,n_epochs=10,val_split=0.2):
    if type(in_shape) == int:
        vae,vae_loss = get_vae(in_shape)
    else:
        vae,vae_loss = get_conv_vae(in_shape)
    model_checkpoint = ModelCheckpoint(save_name, monitor='loss',verbose=1, save_best_only=True)
    vae = load_model(load_name, custom_objects={'latent_dim': latent_dim, 'epsln_std': epsln_std, 'vae_loss': vae_loss})

    ds = HDF5Matrix(hdf_name,ds_name)
    vae.fit(ds, ds, epochs=n_epochs, verbose=1, validation_split=val_split, shuffle='batch', callbacks=[model_checkpoint])
    return vae

def encode_data(model,data,batch_size=32,layer_name='z_mean'):
    intermediate_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    enc = intermediate_model.predict(data, batch_size=batch_size)
    return enc


#%%         U-Net
class Conv_Layer():
    def __init__(self,n_filters,conv_size):
        self.n_filters = n_filters
        self.conv_size = conv_size

layers = [Conv_Layer(64,3),
          Conv_Layer(128,3),
          Conv_Layer(256,3),
          Conv_Layer(512,3),
          Conv_Layer(1024,3)]

def get_unet(in_shape,layers,pool_size=(2,2),
             drop_layers=None,dropout=0.5,mode='regression',
             n_labels=1,init_lr=0.001,output_activation='sigmoid'):
    #layers is a list of Conv_Layer objects corresponding to the number of filters
    #and convolution size for each level of the U-net. I.e. 2 convolutions will
    #be performed one after the other for each layer in the list with a max-
    #pool in between each.
    #drop_layers are the indices of the levels that have dropout
    inputs = Input(in_shape)

    enc_conv_layers = []
    pool_layers = []

    #--  ENCODING LAYERS  --#
    for i in range(len(layers)):
        if i < 1: #for first layer, convolve inputs
            conv = Conv2D(layers[i].n_filters, layers[i].conv_size,
                          activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        else:   #after first layer, convolve output from previous pooling layer
            conv = Conv2D(layers[i].n_filters, layers[i].conv_size,
                          activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool_layers[-1])
        print "conv"+str(i+1)+" shape:",conv.shape
        #convolve the output of the first convolution
        conv = Conv2D(layers[i].n_filters, layers[i].conv_size,
                      activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        #employ dropout on the output of the second convolution if appropriate
        if i in drop_layers:
            conv = Dropout(dropout)(conv)
        #add to list of encoding layers
        enc_conv_layers.append(conv)
        print "conv"+str(i+1)+" shape:",conv.shape
        #for all but the lowest resolution add max-pooling layer
        if i < len(layers) - 1:
            pool = MaxPooling2D(pool_size)(conv)
            pool_layers.append(pool)
            print "pool"+str(i+1)+" shape:",pool.shape


    #--  DECODING LAYERS  --#
    dec_conv_layers = []
    for i in range(1,len(layers)):
        idx = len(layers) - i - 1
        if i == 1:
            up = Conv2D(layers[idx].n_filters, 2, activation = 'relu', \
                        padding = 'same', kernel_initializer = 'he_normal')\
                        (UpSampling2D(size = (2,2))(enc_conv_layers[-1]))
        else:
            up = Conv2D(layers[idx].n_filters, 2, activation = 'relu', \
                        padding = 'same', kernel_initializer = 'he_normal')\
                        (UpSampling2D(size = (2,2))(dec_conv_layers[-1]))
        merged = Concatenate(axis=3)([enc_conv_layers[idx],up])
        conv = Conv2D(layers[idx].n_filters, layers[idx].conv_size,
                      activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merged)
        conv = Conv2D(layers[idx].n_filters, layers[idx].conv_size,
                      activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)

        dec_conv_layers.append(conv)
    outputs = dec_conv_layers[-1]


    #--  DEFINE LOSS AND COMPILE MODEL  --#
    if mode == 'classification':
        #add final classification layer
        if n_labels == 1:
            outputs = Conv2D(filters=n_labels, kernel_size=(1,1),
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv2D(filters=n_labels, kernel_size=(1,1),
                            activation='softmax')(outputs)
        #build model
        unet_model = Model(inputs=inputs, outputs=outputs)
        #compile with appropriate loss
        if n_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error,
                                optimizer=opt.Adam(lr=init_lr),
                                metrics=[dice_coefficient])
        else:
            unet_model.compile(loss='categorical_crossentropy',
                                optimizer=opt.Adam(lr=init_lr),
                                metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        #Add final reconstruction layer
        outputs = Conv2D(1, kernel_size=(1,1),activation=output_activation)(outputs)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr),
                           metrics = ['mean_squared_error',])
    return unet_model

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

