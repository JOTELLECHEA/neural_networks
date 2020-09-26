import pandas as pd
import numpy as np
file = 'testing.csv'
modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
data = pd.read_csv(file)
print(data.to_string(justify='left',columns=modelParam,header=True,index=False))