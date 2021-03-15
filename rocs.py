# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Creates a csv file for ROC plots from h5 file, to then be used by rocplots.py.
###########################################################################################################################
# Import packages.
import sys
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
from sklearn.metrics import auc, roc_curve

sc = StandardScaler()

# Fixed values.
seed = 42
tree = "OutputTree"

# File used.
parser = argparse.ArgumentParser(
    description="Imports weights from trained NN, files located in data/"
)
parser.add_argument("--file", type=str, help="Use '--file' followed by a *.h5 file")
args = parser.parse_args()
file = "data/" + str(args.file)

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
type = ["flav", "pT", "eta", "phi", "b", "c"]
LeptonVAR = []
JetVAR = []
for i in range(4):
    for j in range(3):
        LeptonVAR.append("lepton" + str(j + 1) + type[i])
for i in range(1, 6):
    for j in range(10):
        JetVAR.append("jet" + str(j + 1) + type[i])

# Auto select feature set.
phase = int(input("Enter 1 for High, 2 for Low, or 3 for both:"))

if phase == 1:
    branches = sorted(HighLevel) + ["weights", "truth"]
elif phase == 2:
    branches = sorted(LeptonVAR + JetVAR) + ["weights", "truth"]
elif phase == 3:
    branches = sorted(HighLevel + JetVAR + LeptonVAR) + ["weights", "truth"]
else:
    print("Invalid option")
    sys.exit()

numBranches = len(branches) - 2

# Data read from file.
signal = uproot.open("data/new_TTHH.root")[tree]
df_signal = signal.pandas.df(branches)

bkgTTBB = uproot.open("data/new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)

bkgTTH = uproot.open("data/new_TTH.root")[tree]
df_bkgTTH = bkgTTH.pandas.df(branches)

bkgTTZ = uproot.open("data/new_TTZ.root")[tree]
df_bkgTTZ = bkgTTZ.pandas.df(branches)

df_background = pd.concat([df_bkgTTBB, df_bkgTTH, df_bkgTTZ])

shuffleBackground = shuffle(df_background, random_state=seed)


# signal and limited shuffle background data to counter inbalanced data problem.
rawdata = pd.concat([df_signal, shuffleBackground])

# Drops weights column from rawdata.
X = rawdata.drop("weights", axis=1)

# Transforms X to have a mean = 0 and a variance = 1.
X = sc.fit_transform(X)

# Weights of data applied to scale events.
scalefactor = 0.00232 * 0.608791
sigw = rawdata["weights"][: len(signal)] * scalefactor
bkgw = rawdata["weights"][len(signal) :]

# Labeling data with 1's and 0's to distinguish.
y = np.concatenate((np.ones(len(signal)), np.zeros(len(shuffleBackground))))


neuralNet = keras.models.load_model(file)

y_predicted = neuralNet.predict(X)

# False postive rate, true positive rate and threshold from trained NN model.
fpr, tpr, thresholds = roc_curve(y, y_predicted)

flag2 = 1
if flag2 == 1:
    numbins = 100000
    data = {"fpr": fpr, "tpr": tpr, "bkgR": 1 / fpr}
    df = pd.DataFrame(data)

    # Auto save, Phase intialized in line 30.
    if phase == 1:
        df.to_csv("highlvlvars.csv", mode="a", header=True, index=False)
    elif phase == 2:
        df.to_csv("lowlvlvars.csv", mode="a", header=True, index=False)
    elif phase == 3:
        df.to_csv("highandlowlvlvars.csv", mode="a", header=True, index=False)

""" This code is here until we determine if it is still needed."""

# sigScore = neuralNet.predict(X[y > 0.5]).ravel()
# bkgScore = neuralNet.predict(X[y < 0.5]).ravel()
# sigSUM = len(sigScore)
# bkgSUM = len(bkgScore)

# xlimit = (0, 1)
# tp = []
# fp = []
# hist, bins = np.histogram(sigScore, bins=numbins, range=xlimit, density=False)
# count = 0
# for i in range(numbins - 1, -1, -1):
#     count += hist[i] / sigSUM
#     tp.append(count)
# hist, bins = np.histogram(bkgScore, bins=numbins, range=xlimit, density=False)
# count = 0
# for j in range(numbins - 1, -1, -1):
#     count += hist[j] / bkgSUM
#     fp.append(count)
# area = auc(fp, tp)
# xplot = tp
# yplot = fp
# # computes max signif
# sigSUM = len(sigScore) * scalefactor
# tp = np.array(tp) * sigSUM
# fp = np.array(fp) * bkgSUM
# syst = 0.0
# stat = 0.0
# maxsignif = 0.0
# maxs = 0
# maxb = 0
# bincounter = numbins - 1
# bincountatmaxsignif = 999
# bkgRegection = []
# for t, f in zip(tp, fp):
#     bkgRegection.append(1/f)
#     signif = slug.getZPoisson(t, f, stat, syst)
#     if f >= 10 and signif > maxsignif:
#         maxsignif = signif
#         maxs = t
#         maxb = f
#         bincountatmaxsignif = bincounter
#         score = bincountatmaxsignif / numbins
#     bincounter -= 1
# print(
#     "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n"
#     % (score, maxsignif, maxs, maxb)
# )

# Data to be saved as a csv. Then converted to a pandas data frame.
