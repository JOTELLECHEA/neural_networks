import uproot
import numpy as np
import pandas as pd

tree = "OutputTree" # The name of the tree object in ROOT.


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

### Low Level variables START -
type = ["flav", "pT", "eta", "phi", "b", "c"]
LeptonVAR = []
JetVAR = []
jetcount = 10 # This number can be changed. The range is 0-21. Currnetly only focusing on 10 first 10 jets. 
for i in range(4):
    for j in range(3):
        LeptonVAR.append("lepton" + str(j + 1) + type[i])
for i in range(1, 6):
    for j in range(jetcount):
        JetVAR.append("jet" + str(j + 1) + type[i])
###                                               -END

phase = 3 # Keep at 3 to include all data, modify to limit the data by feature type.
# Auto select feature set.
if phase == 1:
    branches = sorted(HighLevel) + ["weights", "truth"]
elif phase == 2:
    branches = sorted(LeptonVAR + JetVAR) + ["weights", "truth"]
elif phase == 3:
    branches = sorted(HighLevel + JetVAR + LeptonVAR) + ["weights", "truth"]

numBranches = len(branches) - 2 


#### Here is where we will open/read/write the ROOT files. ####

# Data read from file. 
# Filepath doesn't need to be changed if script is ran from slugpu.
signal = uproot.open("/home/jtellece/neural_networks/data/new_TTHH.root")[tree] 

# The [:5] is reading in the first 5 events. 
df_signal = signal.np.arrays(branches, library="pd")[:5] # Using Uproot4 which is installed in slugpu 
# df_signal = signal.pandas.df(branches)[:5] # Using Uproot3
# df_signal.head() # This will print out the first few events in table form. 

# This saves the pandas dataframe above into a csv file. 
# This is optional you can just df_signal as a n-d numpy array. 
df_signal.to_csv('TTHH.csv') 

bkgTTBB = uproot.open("/home/jtellece/neural_networks/data/new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)[:5]
df_bkgTTBB.to_csv('TTBB.csv')


bkgTTH = uproot.open("/home/jtellece/neural_networks/data/new_TTH.root")[tree]
df_bkgTTH = bkgTTH.pandas.df(branches)[:5]
df_bkgTTH.to_csv('TTH.csv')


bkgTTZ = uproot.open("/home/jtellece/neural_networks/data/new_TTZ.root")[tree]
df_bkgTTZ = bkgTTZ.pandas.df(branches)[:5]
df_bkgTTZ.to_csv('TTZ.csv')

print('DONE')