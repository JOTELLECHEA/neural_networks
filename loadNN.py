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

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

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
        "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n"
        % (score, maxsignif, maxs, maxb)
    )
    fig0 = plt.figure(0)
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
    # plt.show()


flag = 1
if flag == 1:
    # signal events
    Sweights = []
    NNsnumjet = []
    NNsnumlep = []
    NNsbtag = []
    NNssrap = []
    NNscent = []
    NNsm_bb = []
    NNsh_b = []
    NNsmt1 = []
    NNsdr1 = []
    NNsjetpt1 = []
    NNsjetpt2 = []
    NNsjetpt3 = []
    NNsjetpt4 = []
    NNsjetpt5 = []
    NNsjetpt6 = []
    NNsjetpt7 = []
    NNsjetpt8 = []
    NNsjetpt9 = []
    NNsjetpt10 = []
    NNsjeteta1 = []
    NNsjeteta2 = []
    NNsjeteta3 = []
    NNsjeteta4 = []
    NNsjeteta5 = []
    NNsjeteta6 = []
    NNsjeteta7 = []
    NNsjeteta8 = []
    NNsjeteta9 = []
    NNsjeteta10 = []
    NNsjetphi1 = []
    NNsjetphi2 = []
    NNsjetphi3 = []
    NNsjetphi4 = []
    NNsjetphi5 = []
    NNsjetphi6 = []
    NNsjetphi7 = []
    NNsjetphi8 = []
    NNsjetphi9 = []
    NNsjetphi10 = []

    # background events
    Bweights = []
    NNbnumjet = []
    NNbnumlep = []
    NNbbtag = []
    NNbsrap = []
    NNbcent = []
    NNbm_bb = []
    NNbh_b = []
    NNbmt1 = []
    NNbdr1 = []
    NNbjetpt1 = []
    NNbjetpt2 = []
    NNbjetpt3 = []
    NNbjetpt4 = []
    NNbjetpt5 = []
    NNbjetpt6 = []
    NNbjetpt7 = []
    NNbjetpt8 = []
    NNbjetpt9 = []
    NNbjetpt10 = []
    NNbjeteta1 = []
    NNbjeteta2 = []
    NNbjeteta3 = []
    NNbjeteta4 = []
    NNbjeteta5 = []
    NNbjeteta6 = []
    NNbjeteta7 = []
    NNbjeteta8 = []
    NNbjeteta9 = []
    NNbjeteta10 = []
    NNbjetphi1 = []
    NNbjetphi2 = []
    NNbjetphi3 = []
    NNbjetphi4 = []
    NNbjetphi5 = []
    NNbjetphi6 = []
    NNbjetphi7 = []
    NNbjetphi8 = []
    NNbjetphi9 = []
    NNbjetphi10 = []

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
                NNsjetpt1.append(rawdata["jet1pT"].values[i])
                NNsjetpt2.append(rawdata["jet2pT"].values[i])
                NNsjetpt3.append(rawdata["jet3pT"].values[i])
                NNsjetpt4.append(rawdata["jet4pT"].values[i])
                NNsjetpt5.append(rawdata["jet5pT"].values[i])
                NNsjetpt6.append(rawdata["jet6pT"].values[i])
                NNsjetpt7.append(rawdata["jet7pT"].values[i])
                NNsjetpt8.append(rawdata["jet8pT"].values[i])
                NNsjetpt9.append(rawdata["jet9pT"].values[i])
                NNsjetpt10.append(rawdata["jet10pT"].values[i])
                NNsjeteta1.append(rawdata["jet1eta"].values[i])
                NNsjeteta2.append(rawdata["jet2eta"].values[i])
                NNsjeteta3.append(rawdata["jet3eta"].values[i])
                NNsjeteta4.append(rawdata["jet4eta"].values[i])
                NNsjeteta5.append(rawdata["jet5eta"].values[i])
                NNsjeteta6.append(rawdata["jet6eta"].values[i])
                NNsjeteta7.append(rawdata["jet7eta"].values[i])
                NNsjeteta8.append(rawdata["jet8eta"].values[i])
                NNsjeteta9.append(rawdata["jet9eta"].values[i])
                NNsjeteta10.append(rawdata["jet10eta"].values[i])
                NNsjetphi1.append(rawdata["jet1phi"].values[i])
                NNsjetphi2.append(rawdata["jet2phi"].values[i])
                NNsjetphi3.append(rawdata["jet3phi"].values[i])
                NNsjetphi4.append(rawdata["jet4phi"].values[i])
                NNsjetphi5.append(rawdata["jet5phi"].values[i])
                NNsjetphi6.append(rawdata["jet6phi"].values[i])
                NNsjetphi7.append(rawdata["jet7phi"].values[i])
                NNsjetphi8.append(rawdata["jet8phi"].values[i])
                NNsjetphi9.append(rawdata["jet9phi"].values[i])
                NNsjetphi10.append(rawdata["jet10phi"].values[i])
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
                NNbjetpt1.append(rawdata["jet1pT"].values[i])
                NNbjetpt2.append(rawdata["jet2pT"].values[i])
                NNbjetpt3.append(rawdata["jet3pT"].values[i])
                NNbjetpt4.append(rawdata["jet4pT"].values[i])
                NNbjetpt5.append(rawdata["jet5pT"].values[i])
                NNbjetpt6.append(rawdata["jet6pT"].values[i])
                NNbjetpt7.append(rawdata["jet7pT"].values[i])
                NNbjetpt8.append(rawdata["jet8pT"].values[i])
                NNbjetpt9.append(rawdata["jet9pT"].values[i])
                NNbjetpt10.append(rawdata["jet10pT"].values[i])
                NNbjeteta1.append(rawdata["jet1eta"].values[i])
                NNbjeteta2.append(rawdata["jet2eta"].values[i])
                NNbjeteta3.append(rawdata["jet3eta"].values[i])
                NNbjeteta4.append(rawdata["jet4eta"].values[i])
                NNbjeteta5.append(rawdata["jet5eta"].values[i])
                NNbjeteta6.append(rawdata["jet6eta"].values[i])
                NNbjeteta7.append(rawdata["jet7eta"].values[i])
                NNbjeteta8.append(rawdata["jet8eta"].values[i])
                NNbjeteta9.append(rawdata["jet9eta"].values[i])
                NNbjeteta10.append(rawdata["jet10eta"].values[i])
                NNbjetphi1.append(rawdata["jet1phi"].values[i])
                NNbjetphi2.append(rawdata["jet2phi"].values[i])
                NNbjetphi3.append(rawdata["jet3phi"].values[i])
                NNbjetphi4.append(rawdata["jet4phi"].values[i])
                NNbjetphi5.append(rawdata["jet5phi"].values[i])
                NNbjetphi6.append(rawdata["jet6phi"].values[i])
                NNbjetphi7.append(rawdata["jet7phi"].values[i])
                NNbjetphi8.append(rawdata["jet8phi"].values[i])
                NNbjetphi9.append(rawdata["jet9phi"].values[i])
                NNbjetphi10.append(rawdata["jet10phi"].values[i])
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

    sjetpt1 = df_signal["jet1pT"].values
    sjetpt2 = df_signal["jet2pT"].values
    sjetpt3 = df_signal["jet3pT"].values
    sjetpt4 = df_signal["jet4pT"].values
    sjetpt5 = df_signal["jet5pT"].values
    sjetpt6 = df_signal["jet6pT"].values
    sjetpt7 = df_signal["jet7pT"].values
    sjetpt8 = df_signal["jet8pT"].values
    sjetpt9 = df_signal["jet9pT"].values
    sjetpt10 = df_signal["jet10pT"].values
    bjetpt1 = df_background["jet1pT"].values
    bjetpt2 = df_background["jet2pT"].values
    bjetpt3 = df_background["jet3pT"].values
    bjetpt4 = df_background["jet4pT"].values
    bjetpt5 = df_background["jet5pT"].values
    bjetpt6 = df_background["jet6pT"].values
    bjetpt7 = df_background["jet7pT"].values
    bjetpt8 = df_background["jet8pT"].values
    bjetpt9 = df_background["jet9pT"].values
    bjetpt10 = df_background["jet10pT"].values

    sjeteta1 = df_signal["jet1eta"].values
    sjeteta2 = df_signal["jet2eta"].values
    sjeteta3 = df_signal["jet3eta"].values
    sjeteta4 = df_signal["jet4eta"].values
    sjeteta5 = df_signal["jet5eta"].values
    sjeteta6 = df_signal["jet6eta"].values
    sjeteta7 = df_signal["jet7eta"].values
    sjeteta8 = df_signal["jet8eta"].values
    sjeteta9 = df_signal["jet9eta"].values
    sjeteta10 = df_signal["jet10eta"].values
    bjeteta1 = df_background["jet1eta"].values
    bjeteta2 = df_background["jet2eta"].values
    bjeteta3 = df_background["jet3eta"].values
    bjeteta4 = df_background["jet4eta"].values
    bjeteta5 = df_background["jet5eta"].values
    bjeteta6 = df_background["jet6eta"].values
    bjeteta7 = df_background["jet7eta"].values
    bjeteta8 = df_background["jet8eta"].values
    bjeteta9 = df_background["jet9eta"].values
    bjeteta10 = df_background["jet10eta"].values

    sjetphi1 = df_signal["jet1phi"].values
    sjetphi2 = df_signal["jet2phi"].values
    sjetphi3 = df_signal["jet3phi"].values
    sjetphi4 = df_signal["jet4phi"].values
    sjetphi5 = df_signal["jet5phi"].values
    sjetphi6 = df_signal["jet6phi"].values
    sjetphi7 = df_signal["jet7phi"].values
    sjetphi8 = df_signal["jet8phi"].values
    sjetphi9 = df_signal["jet9phi"].values
    sjetphi10 = df_signal["jet10phi"].values
    bjetphi1 = df_background["jet1phi"].values
    bjetphi2 = df_background["jet2phi"].values
    bjetphi3 = df_background["jet3phi"].values
    bjetphi4 = df_background["jet4phi"].values
    bjetphi5 = df_background["jet5phi"].values
    bjetphi6 = df_background["jet6phi"].values
    bjetphi7 = df_background["jet7phi"].values
    bjetphi8 = df_background["jet8phi"].values
    bjetphi9 = df_background["jet9phi"].values
    bjetphi10 = df_background["jet10phi"].values

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
        plt.grid()

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

    fig4 = plt.figure(4)
    num = 9
    ax1 = fig4.add_subplot(4, 3, 1)
    hPlot(sjetpt1, bjetpt1, NNsjetpt1, NNbjetpt1, 0, 1e6, 100, branches[num + 5])
    ax2 = fig4.add_subplot(4, 3, 2)
    hPlot(sjetpt2, bjetpt2, NNsjetpt2, NNbjetpt2, 0, 1e6, 100, branches[num + 10])
    ax3 = fig4.add_subplot(4, 3, 3)
    hPlot(sjetpt3, bjetpt3, NNsjetpt3, NNbjetpt3, 0, 1e6, 100, branches[num + 15])
    ax4 = fig4.add_subplot(4, 3, 4)
    hPlot(sjetpt4, bjetpt4, NNsjetpt4, NNbjetpt4, 0, 1e6, 100, branches[num + 20])
    ax5 = fig4.add_subplot(4, 3, 5)
    hPlot(sjetpt5, bjetpt5, NNsjetpt5, NNbjetpt5, 0, 1e6, 100, branches[num + 25])
    ax6 = fig4.add_subplot(4, 3, 6)
    hPlot(sjetpt6, bjetpt6, NNsjetpt6, NNbjetpt6, 0, 1e6, 100, branches[num + 30])
    ax7 = fig4.add_subplot(4, 3, 7)
    hPlot(sjetpt7, bjetpt7, NNsjetpt7, NNbjetpt7, 0, 1e6, 100, branches[num + 35])
    ax8 = fig4.add_subplot(4, 3, 8)
    hPlot(sjetpt8, bjetpt8, NNsjetpt8, NNbjetpt8, 0, 1e6, 100, branches[num + 40])
    ax9 = fig4.add_subplot(4, 3, 9)
    hPlot(sjetpt9, bjetpt9, NNsjetpt9, NNbjetpt9, 0, 1e6, 100, branches[num + 45])
    ax10 = fig4.add_subplot(4, 3, 10)
    hPlot(sjetpt10, bjetpt10, NNsjetpt10, NNbjetpt10, 0, 1e6, 100, branches[num])

    fig5 = plt.figure(5)
    num = 8
    ax1 = fig5.add_subplot(4, 3, 1)
    hPlot(sjeteta1, bjeteta1, NNsjeteta1, NNbjeteta1, -6, 6, 12, branches[num + 5])
    ax2 = fig5.add_subplot(4, 3, 2)
    hPlot(sjeteta2, bjeteta2, NNsjeteta2, NNbjeteta2, -6, 6, 12, branches[num + 10])
    ax3 = fig5.add_subplot(4, 3, 3)
    hPlot(sjeteta3, bjeteta3, NNsjeteta3, NNbjeteta3, -6, 6, 12, branches[num + 15])
    ax4 = fig5.add_subplot(4, 3, 4)
    hPlot(sjeteta4, bjeteta4, NNsjeteta4, NNbjeteta4, -6, 6, 12, branches[num + 20])
    ax5 = fig5.add_subplot(4, 3, 5)
    hPlot(sjeteta5, bjeteta5, NNsjeteta5, NNbjeteta5, -6, 6, 12, branches[num + 25])
    ax6 = fig5.add_subplot(4, 3, 6)
    hPlot(sjeteta6, bjeteta6, NNsjeteta6, NNbjeteta6, -6, 6, 12, branches[num + 30])
    ax7 = fig5.add_subplot(4, 3, 7)
    hPlot(sjeteta7, bjeteta7, NNsjeteta7, NNbjeteta7, -6, 6, 12, branches[num + 35])
    ax8 = fig5.add_subplot(4, 3, 8)
    hPlot(sjeteta8, bjeteta8, NNsjeteta8, NNbjeteta8, -6, 6, 12, branches[num + 40])
    ax9 = fig5.add_subplot(4, 3, 9)
    hPlot(sjeteta9, bjeteta9, NNsjeteta9, NNbjeteta9, -6, 6, 12, branches[num + 45])
    ax10 = fig5.add_subplot(4, 3, 10)
    hPlot(sjeteta10, bjeteta10, NNsjeteta10, NNbjeteta10, -6, 6, 12, branches[num])

    fig6 = plt.figure(6)
    num = 10
    ax1 = fig6.add_subplot(4, 3, 1)
    hPlot(sjetphi1, bjetphi1, NNsjetphi1, NNbjetphi1, -4, 4, 8, branches[num + 5])
    ax2 = fig6.add_subplot(4, 3, 2)
    hPlot(sjetphi2, bjetphi2, NNsjetphi2, NNbjetphi2, -4, 4, 8, branches[num + 10])
    ax3 = fig6.add_subplot(4, 3, 3)
    hPlot(sjetphi3, bjetphi3, NNsjetphi3, NNbjetphi3, -4, 4, 8, branches[num + 15])
    ax4 = fig6.add_subplot(4, 3, 4)
    hPlot(sjetphi4, bjetphi4, NNsjetphi4, NNbjetphi4, -4, 4, 8, branches[num + 20])
    ax5 = fig6.add_subplot(4, 3, 5)
    hPlot(sjetphi5, bjetphi5, NNsjetphi5, NNbjetphi5, -4, 4, 8, branches[num + 25])
    ax6 = fig6.add_subplot(4, 3, 6)
    hPlot(sjetphi6, bjetphi6, NNsjetphi6, NNbjetphi6, -4, 4, 8, branches[num + 30])
    ax7 = fig6.add_subplot(4, 3, 7)
    hPlot(sjetphi7, bjetphi7, NNsjetphi7, NNbjetphi7, -4, 4, 8, branches[num + 35])
    ax8 = fig6.add_subplot(4, 3, 8)
    hPlot(sjetphi8, bjetphi8, NNsjetphi8, NNbjetphi8, -4, 4, 8, branches[num + 40])
    ax9 = fig6.add_subplot(4, 3, 9)
    hPlot(sjetphi9, bjetphi9, NNsjetphi9, NNbjetphi9, -4, 4, 8, branches[num + 45])
    ax10 = fig6.add_subplot(4, 3, 10)
    hPlot(sjetphi10, bjetphi10, NNsjetphi10, NNbjetphi10, -4, 4, 8, branches[num])

    plt.show()
