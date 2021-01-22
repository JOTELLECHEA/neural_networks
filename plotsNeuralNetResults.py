import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# file = 'hyperparameterRecord_v3.csv'
# file = 'fiveLayerDropout.csv'
# file = 'fiveLayerDropout_3.csv'
file = '12.14.2020.csv'

# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
modelParam = [
    "FileName",
    "ConfusionMatrix [TP FP] [FN TN]",
    "Run Time",
    "AUC",
    "Avg.P",
    "Score",
    "Max Signif",
    "nsig",
    "nbkg",
]
data = pd.read_csv(file)
# print(data.to_string(justify='left',columns=modelParam,header=True,index=False))


# index = len(data[:12]) 
# x =  np.arange(3,3+index,1)
# auc  = data['AUC'][:index]
# maxs = data['Max Signif'][:index]
# avgp = data['Avg.P'][:index]
# score = data['Score'][:index]
# runtime = []
# for i in range(0,index):
# 	hh = int(data['Run Time'][:index][i][7:9])
# 	mm = int(data['Run Time'][:index][i][10:12])
# 	ss = int(data['Run Time'][:index][i][13:15])
# 	runtime.append((hh * 3600 + mm * 60 + ss)/60)



# txt1="Variables held constant: Dropout rate = .1, Optimizers = Adam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 75\n\n\n"

# fig1 = plt.figure(1,figsize=(10, 10))
# fig1.suptitle(txt1, fontsize=16)
# ax1 = fig1.add_subplot(2, 2, 1)

# plt.plot(x,auc,'b-', label = 'AUC')
# # plt.plot(x,maxs,'g-', label = 'Max Signif')
# plt.plot(x,avgp,'r-', label = 'Avg Precision')
# plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.title('Metrics vs. Number of Layers')
# plt.legend(loc = 'lower right')
# # plt.grid()
# # plt.show()

# ax2 = fig1.add_subplot(2, 2, 2)
# # plt.plot(x,auc,'b-', label = 'AUC')
# plt.plot(x,maxs,'g-', label = 'Max Signif')
# # plt.plot(x,avgp,'r-', label = 'Avg Precision')
# # plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.ylabel('Sigma')
# plt.title('Maximum significance vs. Number of Layers')
# plt.legend(loc = 'lower right')
# # plt.grid()

# ax3 = fig1.add_subplot(2, 2, 3)
# # plt.plot(x,auc,'b-', label = 'AUC')
# plt.plot(x,runtime,'k-', label = 'Run Time')
# # plt.plot(x,avgp,'r-', label = 'Avg Precision')
# # plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.ylabel('Minutes')
# plt.title('Run Time vs. Number of Layers')
# plt.legend(loc = 'lower right')
# # plt.grid()
# # fig1.text(.5, .05, txt, ha='center')
# plt.show()
# del runtime
# index = len(data[12:])
# x =  np.arange(3,3+index,1)
# auc  = data['AUC'][index:]
# maxs = data['Max Signif'][index:]
# avgp = data['Avg.P'][index:]
# score = data['Score'][index:]
# runtime = []
# for i in range(index,2*index):
#     hh = int(data['Run Time'][index:][i][7:9])
#     mm = int(data['Run Time'][index:][i][10:12])
#     ss = int(data['Run Time'][index:][i][13:15])
#     runtime.append((hh * 3600 + mm * 60 + ss)/60)



# txt2="Variables held constant: Dropout rate = .1, Optimizers = Adam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 63\n\n\n"
# fig2 = plt.figure(2,figsize=(10, 10))
# fig2.suptitle(txt2, fontsize=16)
# ax1 = fig2.add_subplot(2, 2, 1)

# plt.plot(x,auc,'b-', label = 'AUC')
# # plt.plot(x,maxs,'g-', label = 'Max Signif')
# plt.plot(x,avgp,'r-', label = 'Avg Precision')
# plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.title('Metrics vs. Number of Layers')
# plt.legend(loc = 'lower right')
# # plt.grid()
# # plt.show()

# ax2 = fig2.add_subplot(2, 2, 2)
# # plt.plot(x,auc,'b-', label = 'AUC')
# plt.plot(x,maxs,'g-', label = 'Max Signif')
# # plt.plot(x,avgp,'r-', label = 'Avg Precision')
# # plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.ylabel('Sigma')
# plt.title('Maximum significance vs. Number of Layers')
# plt.legend(loc = 'lower right')
# # plt.grid()

# ax3 = fig2.add_subplot(2, 2, 3)
# # plt.plot(x,auc,'b-', label = 'AUC')
# plt.plot(x,runtime,'k-', label = 'Run Time')
# # plt.plot(x,avgp,'r-', label = 'Avg Precision')
# # plt.plot(x,score,'y-', label = 'Score')
# plt.xlabel('Length of Neural Network')
# plt.ylabel('Minutes')
# plt.title('Run Time vs. Number of Layers')
# plt.legend(loc = 'lower right')
# plt.show()
index = len(data)
x = []
for i in range(7):
    x.append(i+3)
    x.append(i+3)
    x.append(i+3)
    x.append(i+3)
    x.append(i+3) 
x = np.array(x)
auc  = data['AUC'][:index]
maxs = data['Max Signif'][:index]
avgp = data['Avg.P'][:index]
score = data['Score'][:index]
runtime = []
for i in range(0,index):
    hh = int(data['Run Time'][:index][i][7:9])
    mm = int(data['Run Time'][:index][i][10:12])
    ss = int(data['Run Time'][:index][i][13:15])
    runtime.append((hh * 3600 + mm * 60 + ss)/60)



# txt1="Variables held constant: Dropout rate = .1, Optimizers = Nadam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 63\n\n\n"
# txt1="Variables held constant: Dropout rate = .1, Optimizers = Nadam,\n Batch Size = 2048, Monitor = val_loss, Patience = 30, features = 63\n\n\n"
# txt1="Variables held constant: Dropout rate = .1, Optimizers = Adam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 63\n\n\n"
# txt1="Variables held constant: Dropout rate = .1, Optimizers = Adam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 75\n\n\n"
txt1="Variables held constant: Dropout rate = .1, Optimizers = Nadam,\n Batch Size = 512, Monitor = val_loss, Patience = 30, features = 75\n\n\n"

fig1 = plt.figure(1,figsize=(10, 10))
fig1.suptitle(txt1, fontsize=16)
ax1 = fig1.add_subplot(2, 2, 1)

plt.plot(x,auc,'b.', label = 'AUC')
# plt.plot(x,maxs,'g-', label = 'Max Signif')
plt.plot(x,avgp,'r.', label = 'Avg Precision')
plt.plot(x,score,'y.', label = 'Score')
plt.xlabel('Length of Neural Network')
plt.title('Metrics vs. Number of Layers')
plt.legend(loc = 'lower right')
# plt.grid()
# plt.show()

ax2 = fig1.add_subplot(2, 2, 2)
# plt.plot(x,auc,'b-', label = 'AUC')
plt.plot(x,maxs,'g.', label = 'Max Signif')
# plt.plot(x,avgp,'r-', label = 'Avg Precision')
# plt.plot(x,score,'y-', label = 'Score')
plt.xlabel('Length of Neural Network')
plt.ylabel('Sigma')
plt.title('Maximum significance vs. Number of Layers')
plt.legend(loc = 'lower right')
# plt.grid()

ax3 = fig1.add_subplot(2, 2, 3)
# plt.plot(x,auc,'b-', label = 'AUC')
plt.plot(x,runtime,'k.', label = 'Run Time')
# plt.plot(x,avgp,'r-', label = 'Avg Precision')
# plt.plot(x,score,'y-', label = 'Score')
plt.xlabel('Length of Neural Network')
plt.ylabel('Minutes')
plt.title('Run Time vs. Number of Layers')
plt.legend(loc = 'upper left')
# plt.grid()
# fig1.text(.5, .05, txt, ha='center')
plt.show()