# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
import uproot
import numpy as np
import tensorflow as tf
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
# Variables
seed = 42
tree = 'OutputTree'
###########################################################################################################################
# Branches names of high/low level variables aka: features.
# branches = ['numjet','numlep']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT',
#'lep3eta','lep3phi','lep3m','mt1','mt2','mt3','dr1','dr2','dr3','btag','cent','srap','m_bb','h_b']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT'
# 'lep3eta','lep3phi','lep3m']
# branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3']
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1']
numofbranches = len(branches)
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

# X = pd.concat([df_signal,shuffleBackground[0:5709]])
X = pd.concat([df_signal,shuffleBackground])
X = sc.fit_transform(X)

# Labeling data with 1's and 0's to distinguish. 
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.5, random_state=seed)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.1,random_state=seed)

fix_imbal = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
fix_imbal = dict(enumerate(fix_imbal))
# NN model defined as a function.
def build_model(lr):
    model = Sequential()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.add(Dense(10, input_dim=numofbranches, activation='relu'))
    model.add(Dense(10 , activation='relu')) #hidden layer
    model.add(Dense(1 , activation='sigmoid'))#output layer
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    return  model
factor = np.exp(np.log(pow(10,6))/500)
learnR = []
Loss   = []
for i in range(1,500):
    lr = pow(factor,i)*10e-6
    # Using model and setting parameters. 
    keras_model = build_model(lr)
    history = keras_model.fit(X_train, y_train,class_weight=fix_imbal,epochs=1, batch_size=570,validation_data=(X_dev, y_dev),verbose=0)
    # Prediction, fpr,tpr and threshold values for ROC.
    y_pred_keras = keras_model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    # AUC
    roc_auc = auc(fpr_keras, tpr_keras)
    loss = keras_model.evaluate(X_test,y_test,verbose=0)[0]
    # print('AUC=',roc_auc,'lr=',lr,'Loss=',loss)
    Loss.append(loss)
    learnR.append(lr)
# print(learnR)
# print(Loss)
plt.plot(learnR, Loss, '-ok')
plt.yscale('log')
plt.xscale('log')
plt.savefig('lrPlot.png')

