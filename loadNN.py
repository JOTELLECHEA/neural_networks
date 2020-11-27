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
file = str(args.file)

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
    for t, f in zip(tp, fp):
        signif = slug.getZPoisson(t, f, stat, syst)
        if f >= 10 and signif > maxsignif:
            maxsignif = signif
            maxs = t
            maxb = f
            bincountatmaxsignif = bincounter
            score = bincountatmaxsignif / numbins
        bincounter -= 1
    print(
        "Score = %6.3f\n, Signif = %5.2f\n, nsig = %d\n, nbkg = %d\n"
        % (score, maxsignif, maxs, maxb)
    )

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
        weights=sigw,
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
        weights=bkgw,
    )
    plt.xlabel("Score")
    plt.ylabel("Distribution")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.subplot(211)
    plt.plot(yplot, xplot, "r-", label="ROC (area = %0.6f)" % (area))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


flag = 1
if flag == 1:

    Sweights = []
    Bweights = []
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

    for i in range(len(X)):
        if i < len(signal):
            if allScore[i] > score:
                NNsnumjet.append(rawdata["numjet"].values[i])
                NNsnumlep.append(rawdata["numlep"].values[i])
                NNsbtag.append(rawdata["btag"].values[i])
                NNssrap.append(rawdata["srap"].values[i])
                NNscent.append(rawdata["cent"].values[i])
                NNsm_bb.append(rawdata["m_bb"].values[i])
                NNsh_b.append(rawdata["h_b"].values[i])
                NNsmt1.append(rawdata["mt1"].values[i])
                NNsdr1.append(rawdata["dr1"].values[i])
                Sweights.append(scalefactor * rawdata["weights"].values[i])
        else:
            if allScore[i] > score:
                NNbnumjet.append(rawdata["numjet"].values[i])
                NNbnumlep.append(rawdata["numlep"].values[i])
                NNbbtag.append(rawdata["btag"].values[i])
                NNbsrap.append(rawdata["srap"].values[i])
                NNbcent.append(rawdata["cent"].values[i])
                NNbm_bb.append(rawdata["m_bb"].values[i])
                NNbh_b.append(rawdata["h_b"].values[i])
                NNbmt1.append(rawdata["mt1"].values[i])
                NNbdr1.append(rawdata["dr1"].values[i])
                Bweights.append(rawdata["weights"].values[i])

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
            weights=bkgw,
        )
        plt.hist(
            x,
            bins=bins,
            histtype="step",
            label="signal",
            linestyle="solid",
            color="firebrick",
            weights=sigw,
        )
        plt.hist(
            ny,
            bins=bins,
            histtype="step",
            label="NN-background",
            linestyle="dashed",
            color="steelblue",
            weights=Bweights,
        )
        plt.hist(
            nx,
            bins=bins,
            histtype="step",
            label="NN-signal",
            linestyle="dashed",
            color="firebrick",
            weights=Sweights,
        )
        plt.legend(loc=1)
        plt.xlabel(Name)
        plt.ylabel("Events")
        plt.yscale("log")

    fig1 = plt.figure(1)

    ax1 = fig1.add_subplot(221)
    hPlot(snumjet, bnumjet, NNsnumjet, NNbnumjet, 1, 21, 21, branches[72])

    ax2 = fig1.add_subplot(222)
    hPlot(snumlep, bnumlep, NNsnumlep, NNbnumlep, 1, 3, 3, branches[73])

    ax3 = fig1.add_subplot(223)
    hPlot(sbtag, bbtag, NNsbtag, NNbbtag, 0, 10, 10, branches[0])

    ax4 = fig1.add_subplot(224)
    hPlot(ssrap, bsrap, NNssrap, NNbsrap, 0, 10, 10, branches[74])

    fig2 = plt.figure(2)

    ax1 = fig2.add_subplot(221)
    hPlot(scent, bcent, NNscent, NNbcent, 0, 1, 10, branches[1])

    ax2 = fig2.add_subplot(222)
    hPlot(sm_bb, bm_bb, NNsm_bb, NNbm_bb, 0, 250, 10, branches[68])

    ax3 = fig2.add_subplot(223)
    hPlot(sh_b, bh_b, NNsh_b, NNbh_b, 0, 1500, 10, branches[5])

    ax4 = fig2.add_subplot(224)
    hPlot(smt1, bmt1, NNsmt1, NNbmt1, 0, 300, 100, branches[69])

    plot3 = plt.figure(3)
    hPlot(sdr1, bdr1, NNsdr1, NNbdr1, 0, 7, 100, branches[2])

    plt.show()
