import pandas as pd
import numpy as np
# pd.options.display.float_format = '${:,.2f}'.format
file = 'hyperparameterRecord.csv'
modelParam  = ['NN Archi.','#Br.','LR','Batch','#Layers','#Neurons','#Epochs','#EpochsRan','AUC','Sigif.','File']
data = pd.read_csv(file)
print(data.to_string(justify='left',columns=modelParam,header=True,index=False))
