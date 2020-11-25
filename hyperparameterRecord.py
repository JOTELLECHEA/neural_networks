import pandas as pd
import numpy as np
# file = 'hyperparameterRecord_v3.csv'
file = 'fiveLayerDropout_1.csv'
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
data = pd.read_csv(file)
print(data.to_string(justify='left',columns=modelParam,header=True,index=True))