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
# alldata        = uproot.open('data/full.root')[tree]
# df_alldata     = alldata.pandas.df(branches) 
shuffleBackground = shuffle(df_background,random_state=seed)
#signal and limited shuffle background data to counter inbalanced data problem.
X = pd.concat([df_signal,shuffleBackground])
z = sc.fit_transform(X)

neuralNet = keras.models.load_model('data.h5')

score = neuralNet.predict(z).ravel()

NNsnumjet = []
NNsnumlep = []
NNsbtag   = []
NNssrap   = []
NNscent   = []
NNsm_bb   = []
NNsh_b    = []
NNsmt1    = []
NNsdr1    = []
NNbnumjet = []
NNbnumlep = []
NNbbtag   = []
NNbsrap   = []
NNbcent   = []
NNbm_bb   = []
NNbh_b    = []
NNbmt1    = []
NNbdr1    = []


for i in range(len(signal)):
    if score[i] >= 1: 
        NNsnumjet.append(X['numjet'].values[i])
        NNsnumlep.append(X['numlep'].values[i])
        NNsbtag.append(X['btag'].values[i])
        NNssrap.append(X['srap'].values[i])
        NNscent.append(X['cent'].values[i])
        NNsm_bb.append(X['m_bb'].values[i])
        NNsh_b.append(X['h_b'].values[i])
        NNsmt1.append(X['mt1'].values[i])
        NNsdr1.append(X['dr1'].values[i])
    # elif score[i] < 0.5:
        NNbnumjet.append(X['numjet'].values[i])
        NNbnumlep.append(X['numlep'].values[i])
        NNbbtag.append(X['btag'].values[i])
        NNbsrap.append(X['srap'].values[i])
        NNbcent.append(X['cent'].values[i])
        NNbm_bb.append(X['m_bb'].values[i])
        NNbh_b.append(X['h_b'].values[i])
        NNbmt1.append(X['mt1'].values[i])
        NNbdr1.append(X['dr1'].values[i])

snumlep = df_signal['numlep'].values
bnumlep = df_background['numlep'].values

snumjet = df_signal['numjet'].values
bnumjet = df_background['numjet'].values

sbtag = df_signal['btag'].values
bbtag = df_background['btag'].values

ssrap = df_signal['srap'].values
bsrap = df_background['srap'].values

scent = df_signal['cent'].values
bcent = df_background['cent'].values

sm_bb = df_signal['m_bb'].values
bm_bb = df_background['m_bb'].values

sh_b = df_signal['h_b'].values
bh_b = df_background['h_b'].values

smt1 = df_signal['mt1'].values
bmt1 = df_background['mt1'].values

sdr1 = df_signal['dr1'].values
bdr1 = df_background['dr1'].values
def hPlot(x,y,nx,ny,s,e,Name):
    bins = np.linspace(s,e,e)
    plt.hist(y, bins=bins,histtype='step',label='background',linestyle='solid',color='steelblue')
    plt.hist(x, bins=bins,histtype='step',label='signal',linestyle='solid',color='firebrick')
    plt.hist(ny, bins=bins,histtype='step',label='NN-background',linestyle='dashed',color='steelblue')
    plt.hist(nx, bins=bins,histtype='step',label='NN-signal',linestyle='dashed',color='firebrick')
    plt.legend()
    plt.xlabel(Name)
    plt.ylabel('Events')
    plt.yscale('log')

plot1 = plt.figure(1)
hPlot(snumjet,bnumjet,NNsnumjet,NNbnumjet,1,16,branches[0])

plot2 = plt.figure(2)
hPlot(snumlep,bnumlep,NNsnumlep,NNbnumlep,1,3,branches[1])

plot3 = plt.figure(3)
hPlot(sbtag,bbtag,NNsbtag,NNbbtag,0,16,branches[2])

plot4 = plt.figure(4)
hPlot(ssrap,bsrap,NNssrap,NNbsrap,0,10,branches[3])

plot5 = plt.figure(5)
hPlot(scent,bcent,NNscent,NNbcent,0,10,branches[4])

plot6 = plt.figure(6)
hPlot(sm_bb,bm_bb,NNsm_bb,NNbm_bb,0,250,branches[5])

plot7 = plt.figure(7)
hPlot(sh_b,bh_b,NNsh_b,NNbh_b,0,1500,branches[6])

plot8 = plt.figure(8)
hPlot(smt1,bmt1,NNsmt1,NNbmt1,0,16,branches[7])

plot9 = plt.figure(9)
hPlot(sdr1,bdr1,NNsdr1,NNbdr1,0,8,branches[8])

plt.show()