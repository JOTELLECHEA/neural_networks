# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that trains and test a Keras NN.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################

import csv,sys
import uproot # Allows loading/saving of ROOT files without ROOT.
import pandas as pd # Dataframe to work with data from uproot.
import numpy as np
from numpy import array
np.set_printoptions(threshold=sys.maxsize)
import shap # Allows a feature importances plot to be created for NN in Keras.
import tensorflow as tf # Backend need to for Keras.
import tkinter as tk # Used to view plots via ssh.
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Format used to view.
import math
import time
from math import log,sqrt
from tensorflow import keras 
from tensorflow.keras import metrics 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout,Activation 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler # Normalized data to range from (0,1)
sc = StandardScaler()
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve,average_precision_score,roc_curve,auc,roc_auc_score,precision_recall_curve
from sklearn.metrics import confusion_matrix
from datetime import datetime
import slug # Library with common functions used in multiple scripts.
# Variables.
tree = 'OutputTree' 
seed = 42 # Random seed number to reproduce results.

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
# signal         = uproot.open('data/new_signal_tthh.root')[tree] #old data sample.
signal         = uproot.open('data/new_signal_v2.root')[tree]
df_signal      = signal.pandas.df(branches) # Adding features(branches) to dataframe.
background     = uproot.open('data/new_background.root')[tree]
df_background  = background.pandas.df(branches) # Adding features(branches) to dataframe.

# The 3 backgrounds are concatenated we shuffle to make sure they are not sorted. 
shuffleBackground = shuffle(df_background,random_state=seed)

# Signal and shuffle background data.
X = pd.concat([df_signal,shuffleBackground])

# Normalized the data with a Gaussian distrubuition with 0 mean and unit variance.
X = sc.fit_transform(X)

# Labeling data with 1's and 0's to distinguish.(1/positve/signal and 0/negative/background)
# Truth Labels.
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))


# Shuffle full data and split into train/test and validation set.
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.001, random_state=seed)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.2,random_state=seed)

##########################################################################################################

def main(LAYER,BATCH):
    learnRate   = 0.0001
    batchSize   = BATCH
    numEpochs   = 2
    '''NN structure ex. [5,5,5,5,1] 4 layers with 5 neurons each and one output layer.
    this method I made to quickly add layers to model. For loop an array with (n-1) layers
    and lastly adds a 1 for the output. Look at build_model() to see how this array is applied.'''
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

    # filename for loadNN.py script
    name = 'data/'+pre + '-rocDataNN-' + suf +'.csv'


    # filename for keras model to be saved as. 
    modelName = 'data/'+ pre + '-neuralNet-' + suf +'.h5' 

    # filename for plots to be identified by saved model. 
    figname = 'data/' +  pre + '-plots-' + suf  

    # ##########################################################################################################
    # NN model defined as a function.

    def build_model():

        #Create a NN model. 
        model = Sequential() # barebones model no layers. 
        opt = keras.optimizers.Nadam(learning_rate=learnRate) # Best option for most NN. 

        # activation function other options possible.
        act = 'relu' # 0 for negative values, linear for nonzero values. 

        # model.add() adds one layer at a time, 1st layer needs input shape, So we pass the 1st element of network.
        model.add(Dense(network[0], input_dim = numBranches)) #  Dense Layers are fully connected and most common.

        # now we will loop and add layers (1,(n-1))
        for i in  range(1,numLayers-2):
            model.add(Dense(network[i] , activation = act)) # Hidden layers. 
            # Turning off nuerons of layer above in loop with probability = 1-r, so r = 0.25, then 75% of nerouns are kept.  
            model.add(Dropout(0.01)) 

        # Last layer needs to have one neuron for a binary classification(BC) which yields from 0 to 1. 
        model.add(Dense(network[-1] , activation  = 'sigmoid')) # Output layer's activation function for BC needs to be sigmoid.

        # Last step is compiling.
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
