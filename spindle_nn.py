from __future__ import division
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from eeg_vis import *
import pandas as pd
import scipy as sp
import scipy.stats

def get_xception(in_shape,trn_conv,from_scratch):
	#adapted from:
	#https://github.com/fchollet/keras/issues/4465
	#Get back the convolutional part of Xception trained on ImageNet
	if from_scratch:
		model = Xception(weights=None, include_top=False)
	else:
		model = Xception(weights='imagenet', include_top=False)
	#model.summary()

	#Create your own input format (here 3x200x200)
	input = Input(in_shape,name = 'image_input')

	#Use the generated model 
	output = model(input)

	if not trn_conv:
		for layer in model.layers:
			layer.trainable = False
	#Add the classification layers 
	x = GlobalAveragePooling2D()(output)
	x = Dense(2, activation='softmax', name='predictions')(x)

	#Create new model 
	my_model = Model(input=input, output=x)

	#Compile model to prepare for training
	my_model.compile(loss='binary_crossentropy', optimizer='SGD')

	# Open the file
	with open('xception_network_summary.txt','w') as fh:
	    #In the summary, weights and layers from Xception part will be hidden, but they will be fit during the training
	    my_model.summary(print_fn=lambda x: fh.write(x + '\n'))

	return my_model

def get_resnet50(in_shape,trn_feat,from_scratch):
	#adapted from:
	#https://github.com/fchollet/keras/issues/4465
	#Get back the convolutional part of Xception trained on ImageNet
	if from_scratch:
		model = ResNet50(weights=None, include_top=False)
	else:
		model = ResNet50(weights='imagenet', include_top=False)
	#model.summary()

	#Create your own input format (here 3x200x200)
	input = Input(in_shape,name = 'image_input')

	#Use the generated model 
	output = model(input)

	if not trn_feat:
		for layer in model.layers:
			layer.trainable = False
	#Add the classification layers 
	x = GlobalAveragePooling2D()(output)
	x = Dense(2, activation='softmax', name='predictions')(x)

	#Create new model 
	my_model = Model(input=input, output=x)

	#Compile model to prepare for training
	my_model.compile(loss='binary_crossentropy', optimizer='SGD')

	# Open the file
	with open('resnet50_network_summary.txt','w') as fh:
	    #In the summary, weights and layers from Xception part will be hidden, but they will be fit during the training
	    my_model.summary(print_fn=lambda x: fh.write(x + '\n'))

	return my_model


def eval_net(model,in_data,labels,outPath,ex_idx):
	pred=model.predict(in_data)
	str_predFile=outPath+'pred/excerpt'+str(ex_idx)+'_predictions.txt'
	pred_file=open(str_predFile,'w')
	pred_file.write('label prediction')
	for p in range(len(pred)):
		pred_file.write('%f %f, %f %f\n' % (labels[p,0],labels[p,1],pred[p,0],pred[p,1]))
	pred_file.close()
	fpr,tpr,auc=get_roc(str_predFile)
	str_rocFile=outPath+'roc/ex'+str(ex_idx)+'_roc.csv'
	rocFile=open(str_rocFile,'w')
	for j in range(len(fpr)):
	        rocFile.write('%f, ' % (fpr[j]))
	rocFile.write('\n')
	for j in range(len(tpr)):
	        rocFile.write('%f, ' % (tpr[j]))
	rocFile.write('\n')
	rocFile.write('%f' % (auc))
	rocFile.close()
	del pred, fpr, tpr, auc
	return

def get_prf(strFile):
	f=pd.read_csv(strFile)
	d=f.as_matrix()
	pred=d[:,1]
	lab=d[:,0]
	lab=np.array([lab[x].split()[0] for x in range(len(lab))],dtype=float)
	pred=np.array([pred[x].split()[0] for x in range(len(pred))],dtype=float)
	pmin=np.min(pred)
	pmax=np.max(pred)
	pmin,pmax=mean_confidence_interval(pred)
	a=np.array(lab,dtype=float)
	all_prec=[]
	all_rec=[]
	all_f1=[]
	eps=np.finfo(np.float32).eps
	test=False
	th=np.mean(pred)
	started=False
	d=1
	while test==False:
		b=np.array([pred[x]>th for x in range(len(pred))],dtype=float)
		prec,rec,f1=calc_f1(a,b)
		th+=d*0.01

		b=np.array([pred[x]>th for x in range(len(pred))],dtype=float)
		tprec,trec,tf1=calc_f1(a,b)
		if tf1 < f1 or np.isnan(tf1):
			if not started:
				d=-1
			else:
				test=True
		started=True
	return prec, rec, f1, th-d*0.01

def print_roc(predFile,fname)
        fpr,tpr,auc=get_roc(predFile)
        f=open(fname,'w')
        f.write('%f'%(fpr[0]))
        for j in range(1,len(fpr)):
                f.write(', %f'%(fpr[j]))
        f.write('\n')
        f.write('%f'%(tpr[0]))
        for j in range(1,len(tpr)):
                f.write(', %f'%(tpr[j]))
        f.write('\n%f\n'%(auc))
        f.close()

def calc_f1(source,target):
	TP=0
	TN=0
	FP=0
	FN=0
	for i in range(len(source)):
	    if source[i]==1:
	        if target[i]==1:
	            TP=TP+1
	        else:
	            FN=FN+1
	    else:
	        if target[i]==0:
	            TN=TN+1
	        else:
	            FP=FP+1

	if (TP+FP) == 0:
		precision=0
	else:
		precision=TP/(TP+FP)
	if (TP+FN) == 0:
		recall=0
	else:
		recall=TP/(TP+FN)
	if (precision+recall) == 0:
		F1 = 0
	else:
		F1=2*(precision*recall)/(precision+recall)
	return precision, recall, F1


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m-2*h,m