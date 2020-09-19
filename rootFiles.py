# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
import uproot
import numpy as np
from numpy import array
import pandas as pd
import math
tree = 'OutputTree'
###########################################################################################################################
# Branches names of high/low level variables aka: features.
# branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1','lep1pT','lep1eta','lep1phi'
# ,'lep2pT','lep2eta','lep2phi','lep3pT','lep3eta','lep3phi']
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1','lep1pT','lep1eta','lep1phi',
'lep2pT','lep2eta','lep2phi','lep3pT','lep3eta','lep3phi','jet1pT','jet1eta','jet1phi','jet1b','jet1c',
'jet2pT','jet2eta','jet2phi','jet2b','jet2c','jet3pT','jet3eta','jet3phi','jet3b','jet3c',
'jet4pT','jet4eta','jet4phi','jet4b','jet4c','jet5pT','jet5eta','jet5phi','jet5b','jet5c',
'jet6pT','jet6eta','jet6phi','jet6b','jet6c','jet7pT','jet7eta','jet7phi','jet7b','jet7c',
'jet8pT','jet8eta','jet8phi','jet8b','jet8c','jet9pT','jet9eta','jet9phi','jet9b','jet9c',
'jet10pT','jet10eta','jet10phi','jet10b','jet10c','jet11pT','jet11eta','jet11phi','jet11b','jet11c',
'jet12pT','jet12eta','jet12phi','jet12b','jet12c','jet13pT','jet13eta','jet13phi','jet13b','jet13c',
'jet14pT','jet14eta','jet14phi','jet14b','jet14c','jet15pT','jet15eta','jet15phi','jet15b','jet15c',
'jet16pT','jet16eta','jet16phi','jet16b','jet16c','jet17pT','jet17eta','jet17phi','jet17b','jet17c',
'jet18pT','jet18eta','jet18phi','jet18b','jet18c','jet19pT','jet19eta','jet19phi','jet19b','jet19c',
'jet20pT','jet20eta','jet20phi','jet20b','jet20c','jet21pT','jet21eta','jet21phi','jet21b','jet21c']
numBranches = len(branches)
###########################################################################################################################
# signal         = uproot.open('data/new_signal_tthh.root')[tree]
# df_signal      = signal.pandas.df(branches)
# background     = uproot.open('data/background.root')[tree]
# df_background  = background.pandas.df(branches)
