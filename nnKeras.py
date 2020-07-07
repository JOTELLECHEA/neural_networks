# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
import numpy as np
import uproot
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

tree = 'OutputTree'

signal  = uproot.open('new_signal_tthh.root')[tree]
bg_ttZ  = uproot.open('new_background_ttZ.root')[tree]
bg_ttbb = uproot.open('new_background_ttbb.root')[tree]
bg_ttH  = uproot.open('new_background_ttH.root')[tree]
background = np.concatenate((bg_ttbb,bg_ttH,bg_ttZ))