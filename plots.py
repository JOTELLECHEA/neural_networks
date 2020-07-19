# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
import keras
from keras import metrics
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt

tree = 'OutputTree'
# branches = ['btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
# branches = ['numjet','numlep']
branches = ['mcweight','numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
numofbranches = len(branches)

signal         = uproot.open('new_signal_tthh.root')[tree]
df_signal      = signal.pandas.df(branches,flatten=False)
background     = uproot.open('background.root')[tree]
df_background  = background.pandas.df(branches,flatten=False)



df_signal[['cent']].plot(color='r',kind='hist',bins=np.linspace(0,1,11))
plt.xlabel('Centrality')
plt.ylabel('Events Not Normalised')

plt.savefig('sig.png')

df_background[['cent']].plot(kind='hist',bins=np.linspace(0,1,11))
plt.xlabel('Centrality')
plt.ylabel('Events Not Normalised')
plt.savefig('back.png')

plt.savefig('test.png')