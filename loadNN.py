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

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
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
    "numjet",
    "numlep",
    "btag",
    "srap",
    "cent",
    "m_bb",
    "h_b",
    "mt1",
    "mt2",
    "mt3",
    "dr1",
    "dr2",
    "dr3",
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
# Constants = ['Hmass','Electronmass','Zmass','Muonmass','Wmass','Taumass','Umass','Dmass','Cmass','Smass','Tmass','Bmass','speedLight']
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
    "weights"
]  # ,'jet11pT','jet11eta',
# 'jet11phi','jet11b','jet11c','jet12pT','jet12eta','jet12phi','jet12b','jet12c','jet13pT','jet13eta','jet13phi','jet13b','jet13c','jet14pT','jet14eta','jet14phi',
# 'jet14b','jet14c','jet15pT','jet15eta','jet15phi','jet15b','jet15c','jet16pT','jet16eta','jet16phi','jet16b','jet16c','jet17pT','jet17eta','jet17phi','jet17b',
# 'jet17c','jet18pT','jet18eta','jet18phi','jet18b','jet18c','jet19pT','jet19eta','jet19phi','jet19b','jet19c','jet20pT','jet20eta','jet20phi','jet20b','jet20c',
# 'jet21pT','jet21eta','jet21phi','jet21b','jet21c']
branches = sorted(HighLevel + JetVar + LeptonVar)
numBranches = len(branches)

parser = argparse.ArgumentParser(description="Plot 1D plots of sig/bac")
parser.add_argument("--file", type=str, help="Use '--file=' followed by a *.h5 file")
args = parser.parse_args()
file = str(args.file)

# Data read from file.
signal = uproot.open("data/new_signal_v2.root")[tree]
df_signal = signal.pandas.df(branches)
background = uproot.open("data/new_background.root")[tree]
df_background = background.pandas.df(branches)
shuffleBackground = shuffle(df_background, random_state=seed)
# signal and limited shuffle background data to counter inbalanced data problem.
rawdata = pd.concat([df_signal, shuffleBackground])

X = rawdata.drop('weights',axis=1)

X = sc.fit_transform(X)

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

y_predicted = neuralNet.predict(X_test)

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
    area = auc(fp,tp)
    plt.subplot(212)
    plt.hist(
        sigScore,
        color="r",
        alpha=0.5,
        range=xlimit,
        bins=100,
        histtype="stepfilled",
        density=False,
        label="Signal Distribution",
    )
    plt.hist(
        bkgScore,
        color="b",
        alpha=0.5,
        range=xlimit,
        bins=100,
        histtype="stepfilled",
        density=False,
        label="Background Distribution",
    )
    plt.xlabel("Score")
    plt.ylabel("Distribution")
    plt.yscale("log")
    plt.legend(loc='upper right')
    plt.subplot(211)
    plt.plot(fp, tp, "r-", label="ROC (area = %0.6f)"%(area))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

flag = 0
if flag == 1:

    NNsnumjet = []
    NNsnumlep = []
    NNsbtag = []
    NNssrap = []
    NNscent = []
    NNsm_bb = []
    NNsh_b = []
    NNsmt1 = []
    NNsdr1 = []
    NNbnumjet = []
    NNbnumlep = []
    NNbbtag = []
    NNbsrap = []
    NNbcent = []
    NNbm_bb = []
    NNbh_b = []
    NNbmt1 = []
    NNbdr1 = []

    ScoreForMaxSignif = 0.9996988

    for i in range(len(X)):
        if i < len(signal):
            if score[i] > ScoreForMaxSignif:
                NNsnumjet.append(X["numjet"].values[i])
                NNsnumlep.append(X["numlep"].values[i])
                NNsbtag.append(X["btag"].values[i])
                NNssrap.append(X["srap"].values[i])
                NNscent.append(X["cent"].values[i])
                NNsm_bb.append(X["m_bb"].values[i])
                NNsh_b.append(X["h_b"].values[i])
                NNsmt1.append(X["mt1"].values[i])
                NNsdr1.append(X["dr1"].values[i])
        else:
            if score[i] > ScoreForMaxSignif:
                NNbnumjet.append(X["numjet"].values[i])
                NNbnumlep.append(X["numlep"].values[i])
                NNbbtag.append(X["btag"].values[i])
                NNbsrap.append(X["srap"].values[i])
                NNbcent.append(X["cent"].values[i])
                NNbm_bb.append(X["m_bb"].values[i])
                NNbh_b.append(X["h_b"].values[i])
                NNbmt1.append(X["mt1"].values[i])
                NNbdr1.append(X["dr1"].values[i])

    snumlep = df_signal["numlep"].values
    bnumlep = df_background["numlep"].values

    snumjet = df_signal["numjet"].values
    bnumjet = df_background["numjet"].values

    sbtag = df_signal["btag"].values
    bbtag = df_background["btag"].values

    ssrap = df_signal["srap"].values
    bsrap = df_background["srap"].values

    scent = df_signal["cent"].values
    bcent = df_background["cent"].values

    sm_bb = df_signal["m_bb"].values
    bm_bb = df_background["m_bb"].values

    sh_b = df_signal["h_b"].values
    bh_b = df_background["h_b"].values

    smt1 = df_signal["mt1"].values
    bmt1 = df_background["mt1"].values

    sdr1 = df_signal["dr1"].values
    bdr1 = df_background["dr1"].values

    def hPlot(x, y, nx, ny, a, b, c, Name):
        bins = np.linspace(a, b, c)
        plt.hist(
            y,
            bins=bins,
            histtype="step",
            label="background",
            linestyle="solid",
            color="steelblue",
        )
        plt.hist(
            x,
            bins=bins,
            histtype="step",
            label="signal",
            linestyle="solid",
            color="firebrick",
        )
        plt.hist(
            ny,
            bins=bins,
            histtype="step",
            label="NN-background",
            linestyle="dashed",
            color="steelblue",
        )
        plt.hist(
            nx,
            bins=bins,
            histtype="step",
            label="NN-signal",
            linestyle="dashed",
            color="firebrick",
        )
        plt.legend(loc=1)
        plt.xlabel(Name)
        plt.ylabel("Events")
        plt.yscale("log")

    fig1 = plt.figure(1)

    ax1 = fig1.add_subplot(221)
    hPlot(snumjet, bnumjet, NNsnumjet, NNbnumjet, 1, 21, 21, branches[0])

    ax2 = fig1.add_subplot(222)
    hPlot(snumlep, bnumlep, NNsnumlep, NNbnumlep, 1, 3, 3, branches[1])

    ax3 = fig1.add_subplot(223)
    hPlot(sbtag, bbtag, NNsbtag, NNbbtag, 0, 10, 10, branches[2])

    ax4 = fig1.add_subplot(224)
    hPlot(ssrap, bsrap, NNssrap, NNbsrap, 0, 10, 10, branches[3])

    fig2 = plt.figure(2)

    ax1 = fig2.add_subplot(221)
    hPlot(scent, bcent, NNscent, NNbcent, 0, 1, 10, branches[4])

    ax2 = fig2.add_subplot(222)
    hPlot(sm_bb, bm_bb, NNsm_bb, NNbm_bb, 0, 250, 10, branches[5])

    ax3 = fig2.add_subplot(223)
    hPlot(sh_b, bh_b, NNsh_b, NNbh_b, 0, 1500, 10, branches[6])

    ax4 = fig2.add_subplot(224)
    hPlot(smt1, bmt1, NNsmt1, NNbmt1, 0, 300, 100, branches[7])

    plot3 = plt.figure(3)
    hPlot(sdr1, bdr1, NNsdr1, NNbdr1, 0, 7, 100, branches[8])

    plt.show()
    plt.savefig("test.png")
