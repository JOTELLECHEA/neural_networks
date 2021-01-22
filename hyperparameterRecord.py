import pandas as pd
import numpy as np
import argparse



# parser = argparse.ArgumentParser(description="Plot 1D plots of sig/bac")
# parser.add_argument("--file", type=str, help="Use '--file=' followed by a *.h5 file")
# args = parser.parse_args()
# file = "data/" + str(args.file)


# file = 'hyperparameterRecord_v3.csv'
# file = 'fiveLayerDropout_2.csv'
# file = 'fiveLayerDropout_3.csv'
file = 'csv/testelep2.csv'
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
modelParam = [
        'FileName',
        "ConfusionMatrix [TP FP] [FN TN]",
        "Run Time",
        "AUC",
        "Avg.P",
        "Score",
        "Max Signif",
        "nsig",
        "nbkg"
    ]
data = pd.read_csv(file)
print(data.to_string(justify='right',columns=modelParam,header=True,index=1))