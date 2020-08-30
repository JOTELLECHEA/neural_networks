# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################\
import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
seed = 42
tree = 'OutputTree'
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1']
numBranches = len(branches)
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

neuralNet = keras.models.load_model('test.h5')

score = neuralNet.predict(X).ravel()

# numjet = []
# numlep = []

# Signal_NN = pd.DataFrame(np.array([]),columns=branches)

# for i in range(len(score)):
#     for col in branches:
#         if score[i] >= 0.5: 
#             Signal_NN[col][i] = df_signal[col][i]
#             # numjet.append(df_signal['numjet'][i])
#             # numjet.append(df_signal['numjet'][i])


# def plotROC(x,y,AUC):
#     # plt.subplot(211)
#     plt.plot(x,y, lw = 1, label = 'ROC (area = %0.6f)'%(AUC))
#     plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6), label = 'Luck')
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc = 'lower right')
#     plt.grid()
#     plt.show()



# plot1 = plt.figure(1)
# df_signal['numlep'].hist(bins=3)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# # plt.legend(loc = 'lower right')

# plot1 = plt.figure(2)
# df_signal['numjet'].hist(bins=13)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# # plt.legend(loc = 'lower right')

# plot1 = plt.figure(3)
# df_signal['cent'].hist(bins=10)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# # plt.legend(loc = 'lower right')

# plt.show()