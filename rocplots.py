# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that creates a ROC plot for Low, High, and Low + High VARS (Background rejection vs signal efficiency). 
##############################################################################################################################
import pandas as pd
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc


hnlv = pd.read_csv('highandlowlvlvars.csv')
f_hnlv = hnlv['xplot']
t_hnlv = hnlv['yplot']
bkgR_hnlv = hnlv['bkgR']
#
llv = pd.read_csv('lowlvlvars.csv')
f_llv = llv['xplot']
t_llv = llv['yplot']
bkgR_llv = llv['bkgR']

#
hlv = pd.read_csv('highlvlvars.csv')
f_hlv = hlv['xplot']
t_hlv = hlv['yplot']
bkgR_hlv = hlv['bkgR']
#
plt.plot(t_hlv,  bkgR_hlv, "r-", label="High lvl, AUC = %0.3f" % (auc(f_hlv,t_hlv)))
plt.plot(t_llv, bkgR_llv, "b-", label="Low lvl, AUC = %0.3f" % (auc(f_llv,t_llv)))
plt.plot(t_hnlv, bkgR_hnlv, "k-", label="All, AUC = %0.3f" % (auc(f_hnlv,t_hnlv)))

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title("Receiver operating characteristic")
plt.legend(loc="upper right")

plt.grid()
plt.show()