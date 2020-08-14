import pandas as pd
import numpy as np


col = ['branchLen','epochs','batchSize','learningRate','valSize','testSize','AUC']

data = pd.DataFrame(np.array([[9,150,570,'Default',0.5,0.1,0.858928],
	[9,150,570,'Default',0.5,0.1,0.858928],
	# [9,150,570,'Default',0.5,0.1,0.858928]
	# [9,150,570,'Default',0.5,0.1,0.858928]
	# [9,150,570,'Default',0.5,0.1,0.858928]
	# [9,150,570,'Default',0.5,0.1,0.858928]
	# [9,150,570,'Default',0.5,0.1,0.858928]
	# [9,150,570,'Default',0.5,0.1,0.858928]
	]),columns=col)
print(data.head())