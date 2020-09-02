# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
import csv,sys
import uproot
import numpy as np
from numpy import array
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import pandas as pd
import math
from math import log,sqrt
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import time
sc = StandardScaler()
# Variables.
seed = 42
tree = 'OutputTree'
###########################################################################################################################
# Branches names of high/low level variables aka: features.
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1']
numBranches = len(branches)
network     = [50,50,50,1]
learnRate   = 0.01
batchSize   = 570
numLayers   = len(network)
numNeurons  = sum(network)
numEpochs   = 150
areaUnderCurve = 0 
###########################################################################################################################
# Data read from file.
signal         = uproot.open('data/new_signal_tthh.root')[tree]
df_signal      = signal.pandas.df(branches)
background     = uproot.open('data/background.root')[tree]
df_background  = background.pandas.df(branches)
alldata        = uproot.open('data/full.root')[tree]
df_alldata     = alldata.pandas.df(branches) 
shuffleBackground = shuffle(df_background,random_state=seed)
# signal and limited shuffle background data to counter inbalanced data problem.
X = pd.concat([df_signal,shuffleBackground])
X = sc.fit_transform(X)

# Labeling data with 1's and 0's to distinguish. 
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.5, random_state=seed)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.1,random_state=seed)

# Fix data imbalance.
fix_imbal = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
fix_imbal = dict(enumerate(fix_imbal))
##########################################################################################################
# NN model defined as a function.
def build_model():
    model = Sequential()
    opt = keras.optimizers.Adam(learning_rate=learnRate)
    model.add(Dense(network[0], input_dim = numBranches, activation='relu'))
    model.add(Dense(network[1] , activation = 'relu'))   #hidden layer.
    model.add(Dense(network[2] , activation = 'relu'))   #hidden layer.
    model.add(Dense(network[3] , activation  = 'sigmoid')) #output layer.
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return  model

# Plot ROC.
def plotROC(x,y,AUC):
    plt.subplot(211)
    plt.plot(x,y, lw = 1, label = 'ROC (area = %0.6f)'%(AUC))
    plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6), label = 'Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = 'lower right')
    plt.grid()
# 
def compare_train_test(kModel, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = neuralNet.predict(X[y>0.5]).ravel()# signal 
        d2 = neuralNet.predict(X[y<0.5]).ravel()# background
        decisions += [d1, d2]
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = array([low,high])
    
    plt.subplot(212)
    plt.hist(decisions[0],color='r', alpha=0.5, range=low_high, bins=bins,histtype='stepfilled', density=True,label='S (train)')
    plt.hist(decisions[1],color='b', alpha=0.5, range=low_high, bins=bins,histtype='stepfilled', density=True,label='B (train)')

    hist, bins = np.histogram(decisions[2],bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

##########################################################################################################
# Using model and setting parameters. 
neuralNet = build_model()

checkPointsCallBack = ModelCheckpoint('temp.h5',save_best_only=True)
earlyStopCallBack = EarlyStopping(patience=10, restore_best_weights=True)
kModel = neuralNet.fit(X_train, y_train,class_weight=fix_imbal
    ,epochs=numEpochs
    ,batch_size=batchSize
    ,validation_data=(X_dev, y_dev)
    ,verbose=1
    ,callbacks=[earlyStopCallBack,])

# Prediction, fpr,tpr and threshold values for ROC.
y_predicted = neuralNet.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)


# AUC
areaUnderCurve = auc = auc(fpr, tpr)

compare_train_test(kModel, X_train, y_train, X_test, y_test)

plotROC(fpr, tpr, auc)
pd.DataFrame(kModel.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

nSig=(5709./20000.)*990
nBG=((320752+3332932+158645)/(610000.+270000.+5900000.))*(5.85e6+612000.+269000)
def getZPoisson(s, b, stat, syst):
    """
    The significance for optimisation.

    s: total number of signal events
    b: total number of background events
    stat: relative MC stat uncertainty for the total bkg. 
    syst: relative syst uncertainty on background

    Note that the function already accounts for the sqrt(b) 
    uncertainty from the data, so we only need to pass in additional
    stat and syst terms.  e.g. the stat term above is really only
    characterizing the uncertainty due to limited MC statistics used
    to estimate the background yield.
    """
    n = s+b

    # this is a relative uncertainty
    sigma = math.sqrt(stat**2+syst**2)

    # turn into the total uncertainty
    sigma=sigma*b

    if s <= 0 or b <= 0:
        return 0

    factor1=0
    factor2=0
    
    if (sigma < 0.01):
        #In the limit where the total BG uncertainty is zero, 
        #this reduces to approximately s/sqrt(b)
        factor1 = n*log((n/b))
        factor2 = (n-b)
    else:
        factor1 = n*log( (n*(b+sigma**2))/((b**2)+n*sigma**2) )
        factor2 = ((b**2)/(sigma**2))*log( 1 + ((sigma**2)*(n-b))/(b*(b+sigma**2)) )
    
    signif=0
    try:
        signif=math.sqrt( 2 * (factor1 - factor2))
    except ValueError:
        signif=0
        
    return signif
signifs=np.array([])
signifs2={}
syst=0.0
stat=0.0
maxsignif=0.0
maxbdt=2
maxs=0
maxb=0
for f,t,bdtscore in zip(fpr,tpr,thresholds):
    s=nSig*t
    b=nBG*f
    n=s+b
    signif = getZPoisson(s,b,stat,syst)
    np.append(signifs,signif)
    signifs2[f]=signif
    if signif>maxsignif:
        maxsignif=signif
        maxbdt=bdtscore
        maxs=s
        maxb=b
    # print "%8.6f %8.6f %5.2f %5.2f %8.6f %8.6f %8.6f %8.6f %8.6f %10d %10d" % ( t, f, signif, s/sqrt(b), d0i, d1i, d2i, d3i, bdtscore, s, b)
print("Score Threshold for Max Sigf. = %6.3f, Max Signif = %5.2f, nsig = %10d, nbkg = %10d" % (maxbdt,maxsignif,maxs,maxb))
pre = time.strftime('%Y_%m_%d_')
suf = time.strftime('_%H.%M.%S')
name = 'data/'+pre + 'rocDataNN' + suf +'.csv'
modelName = 'data/'+pre + 'neuralNet' + suf +'.h5'
modelParam  = ['NN Archi.','#Branch.','LearnRate','BatchSize','#Layers','#Neurons','#Epochs','AUC','MaxSigif.','File']
df = pd.DataFrame(np.array([[network,numBranches,learnRate,batchSize,numLayers,numNeurons,numEpochs,areaUnderCurve,maxsignif,name[5:]]]),columns=modelParam)
df.to_csv('hyperparameterRecord.csv', mode='a', header=True, index=False)
print(df.to_string(justify='left',columns=modelParam, index=False))
#############################
r0  = ['name','var']
r1  = ['fpr',fpr]
r2  = ['tpr',tpr]
r3  = ['thresholds',thresholds]
r4  = ['Max Signif',maxsignif ]
with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(r0)
        writer.writerow(r1)
        writer.writerow(r2)
        writer.writerow(r3)
        writer.writerow(r4)
csvFile.close()
#############################
print('Do you want to save this Model?')

answer = input('Enter S to save: ')
print(answer)
if (answer == 'S' or answer == 's'):
    print('Saving.....')
    neuralNet.save(modelName)
    print('Modeled saved')
else:
    print('Model Not Saved')
