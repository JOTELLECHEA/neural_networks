import matplotlib.pyplot as plt 
import numpy as np


def plotExp(x):
    y = np.exp(x)
    plt.plot(x,y, lw=1, label='Exp')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.grid()
    # plt.savefig('Test.png')
    plt.show()

x = np.linspace(0,10)

plotExp(x)