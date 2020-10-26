import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# file = 'hyperparameterRecord_v3.csv'
file = 'fiveLayerDropout.csv'
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
data = pd.read_csv(file)
print(data.to_string(justify='left',columns=modelParam,header=True,index=False))

# GPU 
# xlen = []
# for i in range(12):
# 	xlen.append(3 + i)
# 	xlen.append(3 + i)
# 	xlen.append(3 + i)
# auc  = data['AUC'][:36]
# maxs = data['Max Signif'][:36]
# avgp = data['Avg.P'][:36]
# score = data['Score'][:36]

# #CPU 
# xlen = []
# for i in range(5):
# 	xlen.append(10 + i)
# 	xlen.append(10 + i)
# 	xlen.append(10 + i)
# auc  = data['AUC'][36:]
# maxs = data['Max Signif'][36:]
# avgp = data['Avg.P'][36:]
# score = data['Score'][36:]


if True == False:

	plt.plot(xlen,auc,'b-', label = 'AUC')
	# plt.plot(xlen,maxs,'g-', label = 'Max Signif')
	plt.plot(xlen,avgp,'r-', label = 'Avg Precision')
	plt.plot(xlen,score,'y-', label = 'Score')
	plt.xlabel('Length of Neural Network')
	# plt.title('Precision/Recall vs. Threshold Curve')
	plt.legend(loc = 'lower right')
	plt.grid()
	plt.show()

	timedate = data['Y/M/D @ H:M']
	new = {}
	# new[0] = timedate[0].replace('_@_','.')

	# new[0] = new[0].replace('_','.')

	# new[0] = new[0].split('.')
	# new[1] = timedate[1].replace('_@_','.')

	# new[1] = new[1].replace('_','.')

	# new[1] = new[1].split('.')

	for i in range(len(data)):
		new[i] = timedate[i].replace('_@_','.')
		new[i] = new[i].replace('_','.')
		new[i] = new[i].split('.')



	# def deltat(a,b):
	# 	dt = {}
	# 	runtime = 0
	# 	for i in range(6):
	# 		dt[i] = int(b[i])-int(a[i])
	# 	if (dt[0] == 0 and dt[1] == 0 and dt[2] == 0):# year and month the same
	# 		runtime = 3600*dt[3] + 60*dt[4] + dt[5]
	# 		runtime = runtime/3600
	# 	print('RunTime =','%3.2f' % (runtime,),'Hours')
	# 	# return runtime



	# # print('0:YYYY','1:MM','2:DD','3:HH','4:mm','5:ss')
	# # deltat(new[0],new[2])

	# for key in new:
	# 	# print(key,key+1)
	# 	deltat(new[key],new[key+1])
