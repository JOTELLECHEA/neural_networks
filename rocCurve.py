# Written By : Jonathan O. Tellechea and Mike Hance 
# Adviser    : Mike Hance, Phd
# Research   : To find the significance from a BDT/NN/DN.
# Description: Uses a csv file to recreate roc and maxs signif. The csv files are in ~/data/.
################################################################################################
# Imported packages.
import csv
import itertools
import math
from math import log,sqrt
import argparse
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from numpy import array

### parser -START
parser = argparse.ArgumentParser(description= 'sigf of BDT/NN/DN')
parser.add_argument("--file", type=str, help= "Use '--file=' followed by a *.csv file")
parser.add_argument("--syst", action='store', default=0.0)
args = parser.parse_args()
file = str(args.file)
df = pd.read_csv(file)
### -END


def data(x):
    return np.array([float(i) for i in df['var'][x][1:-1].split()])

fpr        = data(0)
tpr        = data(1)
thresholds = data(2)

r = 0.00606061
g = 0.0000677966
b = 0.00147541
m = 0.0001111
t = 0.0000764706 


# with 1-lepton filtering from tthhAnalysis
nSig = (426908)*(990/(930000/0.609))
nBG=((3332932/5900000.)*5.85e6) + ((320752/610000.)*612000.) + ((158645/270000.)*269000)



def getZPoisson(s, b, stat, syst):
    """
    The significance for optimisation.

    s: total number of signal events
    b: total number of background events
    stat: relative MC stat uncertainty for the total bkg. 
    syst: relative syst uncertainty on background

    Note that the function already accounts for the sqrt(b) 
    uncertainty from the data, so we only need to pass in additional
    stat and syst terms.  e.g. the stat term above is really only
    characterizing the uncertainty due to limited MC statistics used
    to estimate the background yield.
    """
    n = s+b

    # this is a relative uncertainty
    sigma = math.sqrt(stat**2+syst**2)

    # turn into the total uncertainty
    sigma=sigma*b

    if s <= 0 or b <= 0:
        return 0

    factor1=0
    factor2=0
    
    if (sigma < 0.01):
        #In the limit where the total BG uncertainty is zero, 
        #this reduces to approximately s/sqrt(b)
        factor1 = n*log((n/b))
        factor2 = (n-b)
    else:
        factor1 = n*log( (n*(b+sigma**2))/((b**2)+n*sigma**2) )
        factor2 = ((b**2)/(sigma**2))*log( 1 + ((sigma**2)*(n-b))/(b*(b+sigma**2)) )
    
    signif=0
    try:
        signif=math.sqrt( 2 * (factor1 - factor2))
    except ValueError:
        signif=0
        
    return signif


scanROC=True
if scanROC:
    
    signifs=np.array([])
    signifs2={}
    syst=float(args.syst)
    stat=0.0
    maxsignif=0.0
    maxbdt=2
    maxs=0
    maxb=0
    for f,t,bdtscore in zip(fpr,tpr,thresholds):
        s=nSig*t
        b=nBG*f
        n=s+b
        signif = getZPoisson(s,b,stat,syst)
        np.append(signifs,signif)
        signifs2[f]=signif
        if signif>maxsignif:
            maxsignif=signif
            maxbdt=bdtscore
            maxs=s
            maxb=b
    print(" Score = %6.7f\n Sign. = %4.2f\n nsig. = %d\n nbkg. = %d" % (maxbdt,maxsignif,maxs,maxb))

drawPlots=True
if drawPlots:
    bins =30

    roc_auc = auc(fpr, tpr)
    plt.subplot(211)

    x = np.linspace(0,1,1000)
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.6f)'%(roc_auc))
    #plt.plot((b,b),(0,1),'b--', label='ttH')
    #plt.plot((m,m),(0,1),'m--', label='ttZ')
    # plt.plot((g,g),(0,1),'g--', label='ttbb')
    plt.plot((t,t),(0,1),'k--', label='total')
    plt.plot(x,r + 0*x,linestyle='--',color='r', label='ttHH')
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend(loc="lower right")
    # plt.subplot(212)
    # plt.hist(d0,color='r', alpha=0.5, range=low_high, bins=bins,histtype='stepfilled', density=True,label='S (train)')
    # plt.hist(d1,color='b', alpha=0.5, range=low_high, bins=bins,histtype='stepfilled', density=True,label='B (train)')

    # hist, bins = np.histogram(d2,bins=bins, range=low_high, density=True)
    # scale = len(d2) / sum(hist)
    # err = np.sqrt(hist * scale) / scale
    # width = (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    # hist, bins = np.histogram(d3,bins=bins, range=low_high, density=True)
    # scale = len(d2) / sum(hist)
    # err = np.sqrt(hist * scale) / scale
    # plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    # plt.xlabel("BDT output")
    # plt.ylabel("Arbitrary units")
    # plt.legend(loc='upper left')
    # plt.yscale('log')
    plt.show()
