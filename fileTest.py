import uproot
import numpy as np
import pandas as pd

tree = "OutputTree"
phase = 3

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

### Low Level START -
type = ["flav", "pT", "eta", "phi", "b", "c"]
LeptonVAR = []
JetVAR = []
for i in range(4):
    for j in range(3):
        LeptonVAR.append("lepton" + str(j + 1) + type[i])
for i in range(1, 6):
    for j in range(10):
        JetVAR.append("jet" + str(j + 1) + type[i])
###                                               -END

# Auto select feature set.
if phase == 1:
    branches = sorted(HighLevel) + ["weights", "truth"]
elif phase == 2:
    branches = sorted(LeptonVAR + JetVAR) + ["weights", "truth"]
elif phase == 3:
    branches = sorted(HighLevel + JetVAR + LeptonVAR) + ["weights", "truth"]

numBranches = len(branches) - 2

# Data read from file.
# signal = uproot.open("data/new_TTHH.root")[tree]
# df_signal = signal.pandas.df(branches)

bkgTTBB = uproot.open("data/new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)

bkgTTH = uproot.open("data/new_TTH.root")[tree]
df_bkgTTH = bkgTTH.pandas.df(branches)

bkgTTZ = uproot.open("data/new_TTZ.root")[tree]
df_bkgTTZ = bkgTTZ.pandas.df(branches)

df_background = pd.concat([df_bkgTTBB, df_bkgTTH, df_bkgTTZ])

bkg2 = uproot.open("data/new_bgk2.root")[tree]
df_bkg2 = bkg2.pandas.df(branches)

sizeConcatBkg = len(df_bkg2)
sizeSplitBkg = len(df_background)
flag = (sizeSplitBkg == sizeConcatBkg)

print('Are both samples the same size: ',flag)

count = 0
sample = len(df_bkg2)
if flag == True:
	for j in range(sample):
		if df_bkg2.values[j][5] == df_background.values[j][5]:
				count += 1
# print(df_bkg2.head())
print('ROOT Files Match by',count/sample)