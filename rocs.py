# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that loads NN weights and makes 1D plots that apply NN score for a cut.
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
import slug

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numpy import array
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc

sc = StandardScaler()
seed = 42
tree = "OutputTree"

# Branches names of high/low level variables aka: features.
HighLevel = [
    # "numjet",
    # "numlep",
    # "btag",
    # "srap",
    # "cent",
    # "m_bb",
    # "h_b",
    # "mt1",
    # "mt2",
    # "mt3",
    # "dr1",
    # "dr2",
    # "dr3",
]
LeptonVar = [
    "lepton1pT",
    "lepton1eta",
    "lepton1phi",
    "lepton1flav",
    "lepton2pT",
    "lepton2eta",
    "lepton2phi",
    "lepton2flav",
    "lepton3pT",
    "lepton3eta",
    "lepton3phi",
    "lepton3flav",
]
JetVar = [
    "jet1pT",
    "jet1eta",
    "jet1phi",
    "jet1b",
    "jet1c",
    "jet2pT",
    "jet2eta",
    "jet2phi",
    "jet2b",
    "jet2c",
    "jet3pT",
    "jet3eta",
    "jet3phi",
    "jet3b",
    "jet3c",
    "jet4pT",
    "jet4eta",
    "jet4phi",
    "jet4b",
    "jet4c",
    "jet5pT",
    "jet5eta",
    "jet5phi",
    "jet5b",
    "jet5c",
    "jet6pT",
    "jet6eta",
    "jet6phi",
    "jet6b",
    "jet6c",
    "jet7pT",
    "jet7eta",
    "jet7phi",
    "jet7b",
    "jet7c",
    "jet8pT",
    "jet8eta",
    "jet8phi",
    "jet8b",
    "jet8c",
    "jet9pT",
    "jet9eta",
    "jet9phi",
    "jet9b",
    "jet9c",
    "jet10pT",
    "jet10eta",
    "jet10phi",
    "jet10b",
    "jet10c",
]
branches = sorted(HighLevel + JetVar + LeptonVar + ["weights"])
numBranches = len(branches) - 1

parser = argparse.ArgumentParser(description="Plot 1D plots of sig/bac")
parser.add_argument("--file", type=str, help="Use '--file=' followed by a *.h5 file")
args = parser.parse_args()
file = "data/" + str(args.file)

# Data read from file.
signal = uproot.open("data/new_signal_v2.root")[tree]
df_signal = signal.pandas.df(branches)
background = uproot.open("data/new_background.root")[tree]
df_background = background.pandas.df(branches)
shuffleBackground = shuffle(df_background, random_state=seed)


# signal and limited shuffle background data to counter inbalanced data problem.
rawdata = pd.concat([df_signal, shuffleBackground])

X = rawdata.drop("weights", axis=1)

X = sc.fit_transform(X)

# signal
scalefactor = 0.00232 * 0.608791
sigw = rawdata["weights"][: len(signal)] * scalefactor
bkgw = rawdata["weights"][len(signal) :]

# Labeling data with 1's and 0's to distinguish.
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev, X_eval, y_dev, y_eval = train_test_split(
    X, y, test_size=0.001, random_state=seed
)
X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=seed
)

neuralNet = keras.models.load_model(file)

allScore = neuralNet.predict(X)

flag2 = 1
if flag2 == 1:
    numbins = 100000

    sigScore = neuralNet.predict(X[y > 0.5]).ravel()
    bkgScore = neuralNet.predict(X[y < 0.5]).ravel()
    sigSUM = len(sigScore)
    bkgSUM = len(bkgScore)

    xlimit = (0, 1)
    tp = []
    fp = []
    hist, bins = np.histogram(sigScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for i in range(numbins - 1, -1, -1):
        count += hist[i] / sigSUM
        tp.append(count)
    hist, bins = np.histogram(bkgScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for j in range(numbins - 1, -1, -1):
        count += hist[j] / bkgSUM
        fp.append(count)
    area = auc(fp, tp)
    xplot = tp
    yplot = fp
    # computes max signif
    sigSUM = len(sigScore) * scalefactor
    tp = np.array(tp) * sigSUM
    fp = np.array(fp) * bkgSUM
    syst = 0.0
    stat = 0.0
    maxsignif = 0.0
    maxs = 0
    maxb = 0
    bincounter = numbins - 1
    bincountatmaxsignif = 999
    bkgRegection = []
    for t, f in zip(tp, fp):
        bkgRegection.append(1/f)
        signif = slug.getZPoisson(t, f, stat, syst)
        if f >= 10 and signif > maxsignif:
            maxsignif = signif
            maxs = t
            maxb = f
            bincountatmaxsignif = bincounter
            score = bincountatmaxsignif / numbins
        bincounter -= 1
    print(
        "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n"
        % (score, maxsignif, maxs, maxb)
    )
    data = {'xplot':xplot,'yplot':yplot,'bkgR':bkgRegection,'maxsignif':maxsignif,'area':area}
 

    df = pd.DataFrame(data)

    # df.to_csv("highlvlvars.csv", mode="a", header=True, index=False)
    # df.to_csv("highandlowlvlvars.csv", mode="a", header=True, index=False)
    df.to_csv("lowlvlvars.csv", mode="a", header=True, index=False)

