import pandas as pd
import numpy as np

file = 'hyperparameterRecord.csv'
modelParam  = ['NN Archi.','#Branch.','LearnRate','BatchSize','#Layers','#Neurons','#Epochs','AUC','MaxSigif.','File']
data = pd.read_csv(file)
print(data.to_string(justify='left',columns=modelParam,header=True,index=False))
