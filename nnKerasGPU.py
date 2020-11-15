# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
import csv,sys
import uproot
import numpy as np
import shap
from numpy import array
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
# import tkinter as tk
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import log,sqrt
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle,class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve,average_precision_score,roc_curve,auc,roc_auc_score,precision_recall_curve
from imblearn.tensorflow import balanced_batch_generator
from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix
import time
from datetime import datetime
import slug
sc = StandardScaler()
# Variables.
tree = 'OutputTree'
seed = 42
#######################################################
# Branches names of high/low level variables aka: features.
HighLevel = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
LeptonVar = ['lepton1pT','lepton1eta','lepton1phi','lepton1flav','lepton2pT','lepton2eta','lepton2phi','lepton2flav','lepton3pT','lepton3eta','lepton3phi','lepton3flav']
# Constants = ['Hmass','Electronmass','Zmass','Muonmass','Wmass','Taumass','Umass','Dmass','Cmass','Smass','Tmass','Bmass','speedLight']
JetVar    = ['jet1pT','jet1eta','jet1phi','jet1b','jet1c','jet2pT','jet2eta','jet2phi','jet2b','jet2c','jet3pT','jet3eta','jet3phi','jet3b','jet3c','jet4pT','jet4eta',
'jet4phi','jet4b','jet4c','jet5pT','jet5eta','jet5phi','jet5b','jet5c','jet6pT','jet6eta','jet6phi','jet6b','jet6c','jet7pT','jet7eta','jet7phi','jet7b','jet7c',
'jet8pT','jet8eta','jet8phi','jet8b','jet8c','jet9pT','jet9eta','jet9phi','jet9b','jet9c','jet10pT','jet10eta','jet10phi','jet10b','jet10c']#,'jet11pT','jet11eta',
# 'jet11phi','jet11b','jet11c','jet12pT','jet12eta','jet12phi','jet12b','jet12c','jet13pT','jet13eta','jet13phi','jet13b','jet13c','jet14pT','jet14eta','jet14phi',
# 'jet14b','jet14c','jet15pT','jet15eta','jet15phi','jet15b','jet15c','jet16pT','jet16eta','jet16phi','jet16b','jet16c','jet17pT','jet17eta','jet17phi','jet17b',
# 'jet17c','jet18pT','jet18eta','jet18phi','jet18b','jet18c','jet19pT','jet19eta','jet19phi','jet19b','jet19c','jet20pT','jet20eta','jet20phi','jet20b','jet20c',
# 'jet21pT','jet21eta','jet21phi','jet21b','jet21c']

branches = sorted(HighLevel + JetVar + LeptonVar)
numBranches = len(branches)
###########################################################################################################################
# Data read from file.
# signal         = uproot.open('data/new_signal_tthh.root')[tree]
signal         = uproot.open('data/new_signal_v2.root')[tree]
df_signal      = signal.pandas.df(branches)
background     = uproot.open('data/new_background.root')[tree]
df_background  = background.pandas.df(branches)
shuffleBackground = shuffle(df_background,random_state=seed)

# signal and limited shuffle background data to counter inbalanced data problem.
X = pd.concat([df_signal,shuffleBackground])
print('df_signal[jet1eta][0]',df_signal['jet1eta'][0])
print('')
print('df_background[jet1eta][0]',df_background['jet1eta'][0])
print('')
print('X first background values',X['jet1eta'][len(signal)+1])
print('1st X term', X['jet1eta'])
X = sc.fit_transform(X)
print('1st X term', X[0])

# Labeling data with 1's and 0's to distinguish.
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))


# Shuffle full data and split into train/test and validation set.
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.001, random_state=seed)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.2,random_state=seed)

##########################################################################################################
def main(LAYER,BATCH):
    learnRate   = 0.0001
    batchSize   = BATCH#pow(2,9)# 5:32 6:64 7:128 8:256 9:512 10: 1024 
    numEpochs   = 2
    # areaUnderCurve = 0
    network = []
    numLayers  = LAYER
    neurons = numBranches
    for i in range(numLayers-1):
        network.append(neurons)
    network.append(1)
    print(network)
    numNeurons  = sum(network)
    #####################################################
    startTime = datetime.now()
    pre = time.strftime('%Y_%m_%d')
    suf = time.strftime('%H.%M.%S')

    # filename for maxsignif
    name = 'data/'+pre + '-rocDataNN-' + suf +'.csv'


    # filename for keras model 
    modelName = 'data/'+ pre + '-neuralNet-' + suf +'.h5' 

    # filename for plots 
    figname = 'data/' +  pre + '-plots-' + suf  
    # ###########################################################################################################################
    # # Data read from file.
    # # signal         = uproot.open('data/new_signal_tthh.root')[tree]
    # signal         = uproot.open('data/new_signal_v2.root')[tree]
    # df_signal      = signal.pandas.df(branches)
    # background     = uproot.open('data/new_background.root')[tree]
    # df_background  = background.pandas.df(branches)
    # shuffleBackground = shuffle(df_background,random_state=seed)

    # # signal and limited shuffle background data to counter inbalanced data problem.
    # X = pd.concat([df_signal,shuffleBackground])
    # print('df_signal[jet1eta][0]',df_signal['jet1eta'][0])
    # print(df_background['jet1eta'][:5])
    # print(X['jet1eta'][len(signal):5])

    # print('1st X term', X['jet1eta'])

    # X = sc.fit_transform(X)
    # print('1st X term', X[0])

    # # Labeling data with 1's and 0's to distinguish.
    # y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))


    # # Shuffle full data and split into train/test and validation set.
    # X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.001, random_state=seed)
    # X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.2,random_state=seed)

    # ##########################################################################################################
    # NN model defined as a function.

    def build_model():
        model = Sequential()
        opt = keras.optimizers.Nadam(learning_rate=learnRate)
        act = 'relu'
        model.add(Dense(network[0], input_dim = numBranches))#, activation=act))
        for i in  range(1,numLayers-2):
            model.add(Dense(network[i] , activation = act))   #hidden layer.
            model.add(Dropout(0.01))
            # model.add(BatchNormalization())
        model.add(Dense(network[-1] , activation  = 'sigmoid')) #output layer.
        model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = tf.keras.metrics.Precision())
        return  model


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
    earlyStopCallBack = EarlyStopping(min_delta=0.001,patience=10, restore_best_weights=True)
    kModel = neuralNet.fit(X_train, y_train
        ,epochs=numEpochs
        ,batch_size=batchSize
        ,validation_data=(X_test, y_test)
        ,verbose=1
        ,callbacks=[earlyStopCallBack,checkPointsCallBack])
    ##########################################################################################################

    y_predicted = neuralNet.predict(X_test)
    y_predicted_round = [1 * (x[0]>=0.5) for x in y_predicted]

    # Prediction, fpr,tpr and threshold values for ROC.
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    precision, recall, thresRecall = precision_recall_curve(y_test, y_predicted)

    # AUC
    aucroc = auc(fpr, tpr)
    # plot1 = plt.figure(1)
    # slug.plotROC(fpr, tpr, aucroc)
    # slug.plotPR(precision,recall,thresRecall)
    # compare_train_test(kModel, X_train, y_train, X_test, y_test)
    ##########################################################################
    
    flag = 0
    if flag == 1:
        plot2 = plt.figure(2)
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(neuralNet, background)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
    ###################################################################
    # pd.DataFrame(kModel.history).plot(figsize=(8,5))
    # plt.grid(True)
    # plt.gca().set_ylim(0,1)
    # plt.savefig(figname + 'modelSummary.png')
    ###################################################################
    # computes max signif
    nSig = (426908)*(990/(930000/0.609))
    nBG=((320752+3332932+158645)/(610000.+270000.+5900000.))*(5.85e6+612000.+269000)
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
        signif = slug.getZPoisson(s,b,stat,syst)
        np.append(signifs,signif)
        signifs2[f]=signif
        if signif>maxsignif:
            maxsignif=signif
            maxbdt=bdtscore
            maxs=s
            maxb=b
        # print "%8.6f %8.6f %5.2f %5.2f %8.6f %8.6f %8.6f %8.6f %8.6f %10d %10d" % ( t, f, signif, s/sqrt(b), d0i, d1i, d2i, d3i, bdtscore, s, b)
    print("Score Threshold for Max Signif. = %6.3f, Max Signif = %5.2f, nsig = %10d, nbkg = %10d" % (maxbdt,maxsignif,maxs,maxb))
    runTime = datetime.now() - startTime
    areaUnderCurve = "{:.4f}".format(aucroc)
    maxsignif = "{:5.2f}".format(maxsignif)
    average_precision = average_precision_score(y_test, y_predicted)
    avgPer='{0:0.4f}'.format(average_precision)
    maxbdt='{0:6.3f}'.format(maxbdt)
    maxs='%10d'%(maxs)
    maxb='%10d'%(maxb)
    cm = confusion_matrix(y_test, y_predicted_round)
    modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
    df = pd.DataFrame(np.array([[network,numBranches,learnRate,batchSize,areaUnderCurve,avgPer,runTime,cm,maxbdt,maxsignif,maxs,maxb]]),columns=modelParam)
    df.to_csv('fiveLayerDropout.csv', mode='a', header=False, index=False)
    print(df.to_string(justify='left',columns=modelParam, header=True, index=False))
    print('Saving model.....')
    neuralNet.save(modelName) # Save Model as a HDF5 filein Data folder
    print('Model Saved')
    print('Saving maxsignif.....')
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
    print('Maxsignif Saved')
