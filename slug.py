import tkinter as tk
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plotPR(x,y,t):
    plt.subplot(411)
    plt.plot(t,x[:-1],'b--', label = 'Precision')
    plt.plot(t,y[:-1],'g-', label = 'Recall')
    plt.ylim([0.00, 1.05])
    plt.xlabel('Threshold')
    plt.title('Precision/Recall vs. Threshold Curve')
    plt.legend(loc = 'lower right')
    plt.grid()

def plotROC(x,y,AUC):
    plt.subplot(412)
    plt.plot(x,y, lw = 1, label = 'ROC (area = %0.6f)'%(AUC))
    plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6), label = 'Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = 'lower right')
    plt.grid()

nSig = (426908)*(990/(930000/0.609))
nBG=((320752+3332932+158645)/(610000.+270000.+5900000.))*(5.85e6+612000.+269000)
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
        factor1 = n*math.log((n/b))
        factor2 = (n-b)
    else:
        factor1 = n*math.log( (n*(b+sigma**2))/((b**2)+n*sigma**2) )
        factor2 = ((b**2)/(sigma**2))*math.log( 1 + ((sigma**2)*(n-b))/(b*(b+sigma**2)) )

    signif=0
    try:
        signif=math.sqrt( 2 * (factor1 - factor2))
    except ValueError:
        signif=0

    return signif