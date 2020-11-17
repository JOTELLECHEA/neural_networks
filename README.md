# neural_networks
Research on  ttHH production via NN/Keras

# Using
```bash
$ python -i nnKerasGPU.py
```
1. Programs Descriptions
   - nnKerasGPU.py
     > NN via Keras and tensorflow with GPU.
   - multiNN.py
     > loops over nnKerasGPU.py to test NN. 
   - slug.py
     >Script that computes max significance, ROC and PR plots.
   - loadNN.py
     >Script that creates cuts on signal & background based on score given by NN. 
   - hyperparameterRecord.py
     >Script that uses keeps a record on results.
   - rootroot.py
     >Uses ROOT to produce histogram plots of jet four vectors.
   - rocCurve.py
     >Uses a csv file to recreate roc and maxs signif.
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
  
