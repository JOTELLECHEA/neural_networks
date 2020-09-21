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
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,precision_recall_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import time
sc = StandardScaler()
# Variables.
seed = 42
tree = 'OutputTree'
###########################################################################################################################
# Branches names of high/low level variables aka: features.
HighLevel = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
LeptonVar = ['lepton1pT','lepton1eta','lepton1phi','lepton1flav','lepton2pT','lepton2eta','lepton2phi','lepton2flav','lepton3pT','lepton3eta','lepton3phi','lepton3flav']
Constants = ['Hmass','Electronmass','Zmass','Muonmass','Wmass','Taumass','Umass','Dmass','Cmass','Smass','Tmass','Bmass','speedLight']
JetVar    = ['jet1pT','jet1eta','jet1phi','jet1b','jet1c','jet2pT','jet2eta','jet2phi','jet2b','jet2c','jet3pT','jet3eta','jet3phi','jet3b','jet3c','jet4pT','jet4eta',
'jet4phi','jet4b','jet4c','jet5pT','jet5eta','jet5phi','jet5b','jet5c','jet6pT','jet6eta','jet6phi','jet6b','jet6c','jet7pT','jet7eta','jet7phi','jet7b','jet7c',
'jet8pT','jet8eta','jet8phi','jet8b','jet8c','jet9pT','jet9eta','jet9phi','jet9b','jet9c','jet10pT','jet10eta','jet10phi','jet10b','jet10c']#,'jet11pT','jet11eta',
# 'jet11phi','jet11b','jet11c','jet12pT','jet12eta','jet12phi','jet12b','jet12c','jet13pT','jet13eta','jet13phi','jet13b','jet13c','jet14pT','jet14eta','jet14phi',
# 'jet14b','jet14c','jet15pT','jet15eta','jet15phi','jet15b','jet15c','jet16pT','jet16eta','jet16phi','jet16b','jet16c','jet17pT','jet17eta','jet17phi','jet17b',
# 'jet17c','jet18pT','jet18eta','jet18phi','jet18b','jet18c','jet19pT','jet19eta','jet19phi','jet19b','jet19c','jet20pT','jet20eta','jet20phi','jet20b','jet20c',
# 'jet21pT','jet21eta','jet21phi','jet21b','jet21c']
branches = JetVar + LeptonVar + HighLevel #+ Constants
numBranches = len(branches)
numBranches = len(branches)
network     = [10,10,1]#[150,150,150,150,150,150,1]#[10,10,1]
learnRate   = 0.0001
batchSize   = pow(2,9)# 5:32 6:64 7:128 8:256 9:512 10: 1024 
numLayers   = len(network)
numNeurons  = sum(network)
numEpochs   = 150
areaUnderCurve = 0
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
X = sc.fit_transform(X)

# Labeling data with 1's and 0's to distinguish.
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))


# Shuffle full data and split into train/test and validation set.
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.5, random_state=seed)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size = 0.1,random_state=seed)

# Fix data imbalance.
# fix_imbal = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
# fix_imbal = dict(enumerate(fix_imbal))
##########################################################################################################
# NN model defined as a function.
# def build_model():
#     model = Sequential()
#     opt = keras.optimizers.Adam(learning_rate=learnRate)
#     # act = 'LeakyReLU'
#     act = 'relu'
#     model.add(Dense(network[0], input_dim = numBranches))#, activation=act))
#     model.add(Dense(network[1] , activation = act))   #hidden layer.
#     model.add(Dense(network[2] , activation = act))   #hidden layer.
#     model.add(Dense(network[3] , activation = act))   #hidden layer.
#     model.add(Dense(network[4] , activation = act))   #hidden layer.
#     model.add(Dense(network[5] , activation = act))   #hidden layer.
#     # model.add(LeakyReLU(alpha=0.1))
#     model.add(Dense(network[6] , activation  = 'sigmoid')) #output layer.
#     model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy','AUC'])
#     return  model
areaunderROC = tf.keras.metrics.AUC()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
metrics = [areaunderROC]

def build_model():
    model = Sequential()
    opt = keras.optimizers.Nadam(learning_rate=learnRate)
    
    model.add(Dense(network[0], input_dim = numBranches))
    model.add(Dense(network[1] ))  #hidden layer.
    # model.add(Dense(network[2] ))  #hidden layer.
    # model.add(Dense(network[3] ))   #hidden layer.
    # model.add(Dense(network[4] ))  #hidden layer.
    # model.add(Dense(network[5] ))   #hidden layer.
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(network[2] , activation  = 'sigmoid')) #output layer.
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = metrics)
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
# Plot PR
def plotPR(x,y,t):
    plt.subplot(211)
    plt.plot(t,x[:-1],'b--', label = 'Precision')
    plt.plot(t,y[:-1],'g-', label = 'Recall')
    plt.ylim([0.00, 1.05])
    plt.xlabel('Threshold')
    plt.title('Precision/Recal vs. Threshold Curve')
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
earlyStopCallBack = EarlyStopping(patience=1, restore_best_weights=True)
kModel = neuralNet.fit(X_train, y_train#, class_weight=fix_imbal
    ,epochs=numEpochs
    ,batch_size=batchSize
    ,validation_data=(X_dev, y_dev)
    ,verbose=1
    ,callbacks=[earlyStopCallBack,checkPointsCallBack])

# Prediction, fpr,tpr and threshold values for ROC.
y_predicted = neuralNet.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
precision, recall, thresRecall = precision_recall_curve(y_test, y_predicted)

# AUC
areaUnderCurve = auc = auc(fpr, tpr)
plot1 = plt.figure(1)
# # plotROC(fpr, tpr, auc)
plotPR(precision,recall,thresRecall)
compare_train_test(kModel, X_train, y_train, X_test, y_test)
##########################################################################
flag = 0
if flag == 1:
    plot2 = plt.figure(2)
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(neuralNet, background)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
###################################################################
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
print("Score Threshold for Max Signif. = %6.3f, Max Signif = %5.2f, nsig = %10d, nbkg = %10d" % (maxbdt,maxsignif,maxs,maxb))
EpochsRan = len(kModel.history['loss'])
pre = time.strftime('%Y_%m_%d_')
suf = time.strftime('_%H.%M.%S')
name = 'data/'+pre + 'rocDataNN' + suf +'.csv'
modelName = 'data/'+pre + 'neuralNet' + suf +'.h5'
dateTime = pre + '@' + suf
areaUnderCurve = "{:.6f}".format(areaUnderCurve)
maxsignif = "{:.3f}".format(maxsignif)
average_precision = average_precision_score(y_test, y_predicted)
avgPer='{0:0.2f}'.format(average_precision)
modelParam  = ['NN Archi.','#Br.','LR','Batch','#Layers','AUC','Sigif.','Avg. Precision','Y/M/D @ H:M']
df = pd.DataFrame(np.array([[network,numBranches,learnRate,batchSize,numLayers,areaUnderCurve,maxsignif,avgPer,dateTime]]),columns=modelParam)
df.to_csv('hyperparameterRecord_v2.csv', mode='a', header=False, index=False)
print(df.to_string(justify='left',columns=modelParam, index=False))
# disp = plot_precision_recall_curve(neuralNet, X_test, y_test)
# disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# print(classification_report(y_test,y_predicted))
print('Do you want to save this Model?')
answer = input('Enter S to save: ')
print(answer)
if (answer == 'S' or answer == 's'):
    print('Saving.....')
    neuralNet.save(modelName)
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
    print('Modeled saved')
else:
    print('Model Not Saved')