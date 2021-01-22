# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################\
import uproot
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
seed = 42
tree = 'OutputTree'
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
branches = HighLevel + JetVar + LeptonVar
numBranches = len(branches)
###########################################################################################################################
parser = argparse.ArgumentParser(description= 'Plot 1D plots of sig/bac')
parser.add_argument("--file", type=str, help= "Use '--file=' followed by a *.h5 file")
args = parser.parse_args()
file = str(args.file)
###########################################################################################################################
# Data read from file.
signal         = uproot.open('data/new_signal_v2.root')[tree]
df_signal      = signal.pandas.df(branches)
background     = uproot.open('data/new_background.root')[tree]
df_background  = background.pandas.df(branches)
shuffleBackground = shuffle(df_background,random_state=seed)
#signal and limited shuffle background data to counter inbalanced data problem.
X = pd.concat([df_signal,shuffleBackground])
z = sc.fit_transform(X)

neuralNet = keras.models.load_model(file)

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

ScoreForMaxSignif = 0.999698937000000
count = 0

for i in range(len(X)):
    if i<len(signal):
        if score[i]>ScoreForMaxSignif:
            count +=1 
            NNsnumjet.append(X['numjet'].values[i])
            NNsnumlep.append(X['numlep'].values[i])
            NNsbtag.append(X['btag'].values[i])
            NNssrap.append(X['srap'].values[i])
            NNscent.append(X['cent'].values[i])
            NNsm_bb.append(X['m_bb'].values[i])
            NNsh_b.append(X['h_b'].values[i])
            NNsmt1.append(X['mt1'].values[i])
            NNsdr1.append(X['dr1'].values[i])
    else:
        if score[i]>ScoreForMaxSignif:
            NNbnumjet.append(X['numjet'].values[i])
            NNbnumlep.append(X['numlep'].values[i])
            NNbbtag.append(X['btag'].values[i])
            NNbsrap.append(X['srap'].values[i])
            NNbcent.append(X['cent'].values[i])
            NNbm_bb.append(X['m_bb'].values[i])
            NNbh_b.append(X['h_b'].values[i])
            NNbmt1.append(X['mt1'].values[i])
            NNbdr1.append(X['dr1'].values[i])