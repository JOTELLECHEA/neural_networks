# neural_networks
Research on  ttHH production via NN/Keras

# TBD

1. Programs Descriptions
   - nnKerasGPU.py
     > NN via Keras and tensorflow with GPU.
     ```bash
     $ python -i nnKerasGPU.py
     >>> main(5,512)
      Script starting....
      [75, 75, 75, 75, 1]
     ```
   - multiNN.py
     > loops over nnKerasGPU.py to test NN. 
   - slug.py
     >Script that computes max significance, ROC and PR plots.
   - loadNN.py
     >Script that creates cuts on signal & background based on score given by NN. 
     ```bash
     $ python -i loadNN.py --file filename.h5
     ```
   - hyperparameterRecord.py
     >Script that uses keeps a record on results.
     ```bash
     $ python hyperparameterRecord.py
         NN Archi.             #Br.  LR      Batch  AUC     Avg.P  Run Time                ConfusionMatrix [TP FP] [FN TN]       Score  Max Signif  nsig  nbkg
     0    [75, 75, 75, 75, 1]  75    0.0001   512   0.9454  0.7637  0 days 00:12:27.407433  [[752999   8610]\n [ 36923  48468]]  0.870   1.00        98   9640
     1    [75, 75, 75, 75, 1]  75    0.0001   512   0.9485  0.7744  0 days 00:18:14.872631  [[754118   7491]\n [ 37182  48209]]  0.865   1.06        89   7116
     2    [75, 75, 75, 75, 1]  75    0.0001   512   0.9474  0.7712  0 days 00:38:35.970770  [[754821   6788]\n [ 38098  47293]]  0.930   1.04        44   1828
     3    [75, 75, 75, 75, 1]  75    0.0001   512   0.9475  0.7718  0 days 00:55:50.455564  [[755124   6485]\n [ 38431  46960]]  0.890   1.05        90   7444
     4    [75, 75, 75, 75, 1]  75    0.0001   512   0.9454  0.7646  0 days 01:05:58.572270  [[753322   8287]\n [ 36996  48395]]  0.877   0.99        92   8567
     5    [75, 75, 75, 75, 1]  75    0.0001   256   0.9484  0.7729  0 days 00:27:49.218513  [[754273   7336]\n [ 37228  48163]]  0.875   1.02        87   7310
     ```
   - rootroot.py
     >Uses ROOT to produce histogram plots of jet four vectors.
   - rocCurve.py
     >Uses a csv file to recreate roc and maxs signif.
     ```bash
     $ python rocCurve.py --file data/2020_11_14-rocDataNN-22.13.33.csv
     Score = 0.8901054
     Sign. = 0.88
     nsig. = 56
     nbkg. = 4074
     ```
2. Files required
   - requirements.txt
      > required progrmas needed to run scripts. 
<!--2. Programs Parser Variables/ Outputs
  - lepvec_Pt.py
     >`--x=i` where  i = 1-4: ttHH,ttbb,ttH,ttZ.-->
     
 <!-- #   >`--help` brings up help.
  # - chisquare.py
 #    >N/A
  # - MVA.py
  #   >`--branch=i` where i = phase1-4
     
<!-- #    >Output file is `ROC_data_file.csv'.
 #  - RocCurve.py
  #   >`--file=i` where  i = ROC_data_phase1-4.csv.

<!-- #   >Output is tmp and can be save in format that is needed.
#   - add_SF_branches.py
 #    >`--file='****.root'`.
     
<!--   #  >`--help` brings up help.
     
 <!-- #   >Creates new ROOT file as `new_****.root`.-->
  
