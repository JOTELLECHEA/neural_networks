import pandas as pd
import numpy as np


# col = ['branchLen','epochs','batchSize','learningRate','valSize','testSize','AUC']

# data_10_10_1 = pd.DataFrame(np.array([[9,150,570,'Default',0.5,0.1,0.858928],
#     [9,150,570,0.01,0.5,0.1,0.849799],
#     [9,150,570,0.02,0.5,0.1,0.852697],
#     [9,150,570,0.03,0.5,0.1,0.50000],
#     [9,150,570,0.02,0.5,0.1,0.849159],
#     [9,150,570,0.02,0.5,0.1,0.851267],
#     [9,150,570,0.01,0.5,0.1,0.846555],#CPU no GPU
#     # [9,150,570,0.01,0.5,0.1,0.849799],
#     ]),columns=col)

# print('\n Using an NN Architecture of 10-10-1\n')
# print(data_10_10_1.head(10)) 

# data_10_10_10_1 = pd.DataFrame(np.array([[9,150,570,0.01,0.5,0.1,0.857230],
#     # [9,150,570,0.01,0.5,0.1,0.849799],
#     # [9,150,570,0.02,0.5,0.1,0.852697],
#     # [9,150,570,0.03,0.5,0.1,0.50000],
#     # [9,150,570,0.02,0.5,0.1,0.849159],
#     # [9,150,570,0.02,0.5,0.1,0.851267],
#     # [9,150,570,0.01,0.5,0.1,0.846555],#CPU no GPU
#     # [9,150,570,0.01,0.5,0.1,0.849799],
#     ]),columns=col)

# print('\n Using an NN Architecture of 10-10-10-1\n')
# print(data_10_10_10_1.head(10)) 
modelParam  = ['Number of Branches','Learning Rate','Batch Size','Number of Layers','Number of Neurons','NN Architecture','Numer of Epochs','AUC']
data = pd.read_csv('hyperparameterRecord.csv')
print(data.to_string(columns=modelParam, index=False))