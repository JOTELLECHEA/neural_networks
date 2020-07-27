# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
import keras
from keras import metrics
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt

tree = 'OutputTree'
branches = ['numjet','numlep','btag','srap','m_bb','mt1','mt2','mt3']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT',
            # 'lep3eta','lep3phi','lep3m','mt1','mt2','mt3','dr1','dr2','dr3','btag','cent','srap','m_bb','h_b']
# branches = ['numlep','numjet','lep1pT','lep1eta','lep1phi','lep1m','lep2pT','lep2eta','lep2phi','lep2m','lep3pT',
# 'lep3eta','lep3phi','lep3m','mt1','mt2','mt3','dr1','dr2','dr3']
# branches = ['btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
# branches = ['numjet','numlep']
# branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','mt2','mt3','dr1','dr2','dr3']
numofbranches = len(branches)

signal         = uproot.open('new_signal_tthh.root')[tree]
df_signal      = signal.pandas.df(branches,flatten=False)
background     = uproot.open('background.root')[tree]
df_background  = background.pandas.df(branches,flatten=False)
data           = uproot.open('full.root')[tree]
df_data        = data.pandas.df(branches,flatten=False) 

X = df_data.values
y = np.concatenate((np.ones(df_signal.shape[0]), np.zeros(df_background.shape[0])))
# X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size = 0.10, random_state=42)
# X_train,X_test, y_train,y_test, = train_test_split(X_dev, y_dev, test_size = 0.33,random_state=42)
X_train,X_test, y_train,y_test, = train_test_split(X, y, test_size = 0.33,random_state=42)

m1 = False
if m1 == True:
    model = Sequential()
    model.add(Dense(numofbranches, input_dim=numofbranches, activation='relu'))
    model.add(Dense(10 , activation='relu'))
    model.add(Dense(1 ,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=1)
    scores = model.evaluate(X_train,y_train)
    # print('\n%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))

    from sklearn.metrics import roc_curve
    y_pred_keras = model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

    print("Area under ROC curve: %.4f"%(roc_auc_score(y_test,y_pred_keras)))


m2 = True
if m2 == True:
    def build_model():
        model = Sequential()
        model.add(Dense(numofbranches, input_dim=numofbranches, activation='relu'))
        model.add(Dense(10 , activation='relu')) #hidden layer
        model.add(Dense(1 ,activation='sigmoid'))#output layer
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        return  model

    from keras.wrappers.scikit_learn import KerasClassifier
    keras_model = build_model()
    keras_model.fit(X_train, y_train, epochs=10, validation_split=0.1)


    from sklearn.metrics import roc_curve
    y_pred_keras = keras_model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

    print("Area under ROC curve: %.4f"%(roc_auc_score(y_test,y_pred_keras)))

    roc_auc = auc(fpr_keras, tpr_keras)
    plt.plot(fpr_keras, tpr_keras, lw=1, label='ROC (area = %0.6f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    plt.savefig('roc-curve.png')
