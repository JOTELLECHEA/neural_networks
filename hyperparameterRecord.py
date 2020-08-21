import pandas as pd
import numpy as np


col = ['branchLen','epochs','batchSize','learningRate','valSize','testSize','AUC']

data = pd.DataFrame(np.array([[9,150,570,'Default',0.5,0.1,0.858928],
	[9,150,570,0.01,0.5,0.1,0.849799],
	[9,150,570,0.02,0.5,0.1,0.852697],
	[9,150,570,0.03,0.5,0.1,0.500000],
	# [9,150,570,0.01,0.5,0.1,0.849799],
	# [9,150,570,0.01,0.5,0.1,0.849799],
	# [9,150,570,0.01,0.5,0.1,0.849799],
	# [9,150,570,0.01,0.5,0.1,0.849799],

	]),columns=col)
print('\n Using an NN Architecture of 10-10-1\n')
print(data.head())