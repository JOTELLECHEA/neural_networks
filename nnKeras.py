import numpy as np
import uproot
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

events = uproot.open('new_signal_tthh.root')['OutputTree']
events.keys()

