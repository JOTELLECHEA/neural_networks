# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
import keras
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc


tree = 'OutputTree'
branches = ['btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
# branches = ['njet','btag','srap']

signal         = uproot.open('new_signal_tthh.root')[tree]
df_signal      = signal.pandas.df(branches,flatten=False)
background     = uproot.open('background.root')[tree]
df_background  = background.pandas.df(branches,flatten=False)
data 		   = uproot.open('full_data.root')[tree]
df_data		   = data.pandas.df(branches,flatten=False)	

X = df_data.values
Y = np.concatenate((np.ones(df_signal.shape[0]), np.zeros(df_background.shape[0])))
X_dev,X_eval, y_dev,y_eval = train_test_split(X, Y, test_size = 0.10, random_state=42)
X_train,X_test, y_train,y_test, = train_test_split(X_dev, y_dev, test_size = 0.5,random_state=42)

# dataset = df_data.values 

# X = dataset[:,0:2]
# Y = dataset[:,2]

model = keras.models.Sequential()
model.add(Dense(120, input_dim=11, activation='relu'))
model.add(Dense(2 , activation='relu'))
model.add(Dense(1 ,activation='sigmoid'))
# model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X,Y, epochs=2)
scores = model.evaluate(X,Y)
print('\n%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))
