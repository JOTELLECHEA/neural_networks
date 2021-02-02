# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: This will repeat nnKerasGPU.py multiple times to run it multiple times unsupervised. Must be edited to run scenario of interest.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
# Imported packages.
import nnKerasGPU as nn

batch = 512

i =5
nn.main(i,batch,0.5)
nn.main(i,256,0.5)
nn.main(i,128,0.5)
nn.main(i,64,0.5)
nn.main(i,32,0.5)
# nn.main(i,batch,0.4)
# nn.main(i,batch,0.35)
# nn.main(i,batch,0.3)
# nn.main(i,batch,0.25)
# nn.main(i,batch,0.2)
# nn.main(i,batch,0.15)
# nn.main(i,batch,0.1)
# nn.main(i,batch,0.05)



