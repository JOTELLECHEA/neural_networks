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
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential,model_from_json,load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
sc = StandardScaler()
# Variables.
seed = 42
tree = 'OutputTree'
name = 'rocDataNN.csv'
###########################################################################################################################
# Branches names of high/low level variables aka: features.
# branches = ['numjet']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT',
#'lep3eta','lep3phi','lep3m','mt1','mt2','mt3','dr1','dr2','dr3','btag','cent','srap','m_bb','h_b']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT'
# 'lep3eta','lep3phi','lep3m']
# branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3']
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1']
numBranches = len(branches)
network     = [10,10,10,1]
learnRate   = 0.01
batchSize   = 570
numLayers   = len(network)
numNeurons  = sum(network)
numEpochs   = 3
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
kModel = neuralNet.fit(X_train, y_train,class_weight=fix_imbal,epochs=numEpochs, batch_size=batchSize,validation_data=(X_dev, y_dev),verbose=1)

# Prediction, fpr,tpr and threshold values for ROC.
y_predicted = neuralNet.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)


# AUC
areaUnderCurve = auc = auc(fpr, tpr)
modelParam  = ['Number of Branches','Learning Rate','Batch Size','Number of Layers','Number of Neurons','NN Architecture','Numer of Epochs','AUC']
df = pd.DataFrame(np.array([[numBranches,learnRate,batchSize,numLayers,numNeurons,network,numEpochs,areaUnderCurve]]),columns=modelParam)
df.to_csv('hyperparameterRecord.csv', mode='a', header=False, index=False)

compare_train_test(kModel, X_train, y_train, X_test, y_test)

print(df.to_string(columns=modelParam, index=False))
plotROC(fpr, tpr, auc)
pd.DataFrame(kModel.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#############################
r0  = ['name','var']
r1  = ['fpr',fpr]
r2  = ['tpr',tpr]
r3  = ['thresholds',thresholds]
with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(r0)
        writer.writerow(r1)
        writer.writerow(r2)
        writer.writerow(r3)
csvFile.close()
#############################
print('Do you want to save this Model?')

answer = input('Enter y or n: ')
print(answer)
if (answer == 'Y' or answer == 'y'):
    print('Saving.....')
    neuralNet.save('test.h5')
else:
    print('DONE')
