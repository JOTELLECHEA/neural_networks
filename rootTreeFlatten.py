#!/usr/bin/env python2

import os, time, sys, argparse,math

import numpy as np
from array import array
import shutil
import random


import ROOT
def augment_rootfile(filepath):

    shutil.copyfile(filepath,"new_"+filepath)

    # get tree to loop over
    treename = "OutputTree"
    t = ROOT.TFile("new_"+filepath, "UPDATE")
    tree = t.Get(treename)

    # define branches
    numlep  = array( 'f', [ 0 ] )
    numjet  = array( 'f', [ 0 ] )
    lep1pT  = array( 'f', [ 0 ] )
    lep1eta = array( 'f', [ 0 ] )
    lep1phi = array( 'f', [ 0 ] )
    lep2pT  = array( 'f', [ 0 ] )
    lep2eta = array( 'f', [ 0 ] )
    lep2phi = array( 'f', [ 0 ] )
    lep3pT  = array( 'f', [ 0 ] )
    lep3eta = array( 'f', [ 0 ] )
    lep3phi = array( 'f', [ 0 ] )
    mt1     = array( 'f', [ 0 ] )
    mt2     = array( 'f', [ 0 ] )
    mt3     = array( 'f', [ 0 ] )
    dr1     = array( 'f', [ 0 ] )
    dr2     = array( 'f', [ 0 ] )
    dr3     = array( 'f', [ 0 ] )
    btag    = array( 'f', [ 0 ] )
    srap    = array( 'f', [ 0 ] )
    cent    = array( 'f', [ 0 ] )
    m_bb    = array( 'f', [ 0 ] )
    h_b     = array( 'f', [ 0 ] )
    chi     = array( 'f', [ 0 ] )
    jet1pT  = array( 'f', [ 0 ] )
    jet1eta = array( 'f', [ 0 ] )
    jet1phi = array( 'f', [ 0 ] )
    jet1b   = array( 'f', [ 0 ] )
    jet1c   = array( 'f', [ 0 ] )
    jet2pT  = array( 'f', [ 0 ] )#
    jet2eta = array( 'f', [ 0 ] )
    jet2phi = array( 'f', [ 0 ] )
    jet2b   = array( 'f', [ 0 ] )
    jet2c   = array( 'f', [ 0 ] )
    jet3pT  = array( 'f', [ 0 ] )#
    jet3eta = array( 'f', [ 0 ] )
    jet3phi = array( 'f', [ 0 ] )
    jet3b   = array( 'f', [ 0 ] )
    jet3c   = array( 'f', [ 0 ] )
    jet4pT  = array( 'f', [ 0 ] )#
    jet4eta = array( 'f', [ 0 ] )
    jet4phi = array( 'f', [ 0 ] )
    jet4b   = array( 'f', [ 0 ] )
    jet4c   = array( 'f', [ 0 ] )
    jet5pT  = array( 'f', [ 0 ] )#
    jet5eta = array( 'f', [ 0 ] )
    jet5phi = array( 'f', [ 0 ] )
    jet5b   = array( 'f', [ 0 ] )
    jet5c   = array( 'f', [ 0 ] )
    jet6pT  = array( 'f', [ 0 ] )#
    jet6eta = array( 'f', [ 0 ] )
    jet6phi = array( 'f', [ 0 ] )
    jet6b   = array( 'f', [ 0 ] )
    jet6c   = array( 'f', [ 0 ] )
    jet7pT  = array( 'f', [ 0 ] )#
    jet7eta = array( 'f', [ 0 ] )
    jet7phi = array( 'f', [ 0 ] )
    jet7b   = array( 'f', [ 0 ] )
    jet7c   = array( 'f', [ 0 ] )
    jet8pT  = array( 'f', [ 0 ] )#
    jet8eta = array( 'f', [ 0 ] )
    jet8phi = array( 'f', [ 0 ] )
    jet8b   = array( 'f', [ 0 ] )
    jet8c   = array( 'f', [ 0 ] )
    jet9pT  = array( 'f', [ 0 ] )#
    jet9eta = array( 'f', [ 0 ] )
    jet9phi = array( 'f', [ 0 ] )
    jet9b   = array( 'f', [ 0 ] )
    jet9c   = array( 'f', [ 0 ] )
    jet10pT  = array( 'f', [ 0 ] )#
    jet10eta = array( 'f', [ 0 ] )
    jet10phi = array( 'f', [ 0 ] )
    jet10b   = array( 'f', [ 0 ] )
    jet10c   = array( 'f', [ 0 ] )
    jet11pT  = array( 'f', [ 0 ] )#
    jet11eta = array( 'f', [ 0 ] )
    jet11phi = array( 'f', [ 0 ] )
    jet11b   = array( 'f', [ 0 ] )
    jet11c   = array( 'f', [ 0 ] )
    jet12pT  = array( 'f', [ 0 ] )#
    jet12eta = array( 'f', [ 0 ] )
    jet12phi = array( 'f', [ 0 ] )
    jet12b   = array( 'f', [ 0 ] )
    jet12c   = array( 'f', [ 0 ] )
    jet13pT  = array( 'f', [ 0 ] )#
    jet13eta = array( 'f', [ 0 ] )
    jet13phi = array( 'f', [ 0 ] )
    jet13b   = array( 'f', [ 0 ] )
    jet13c   = array( 'f', [ 0 ] )
    jet14pT  = array( 'f', [ 0 ] )#
    jet14eta = array( 'f', [ 0 ] )
    jet14phi = array( 'f', [ 0 ] )
    jet14b   = array( 'f', [ 0 ] )
    jet14c   = array( 'f', [ 0 ] )
    jet15pT  = array( 'f', [ 0 ] )#
    jet15eta = array( 'f', [ 0 ] )
    jet15phi = array( 'f', [ 0 ] )
    jet15b   = array( 'f', [ 0 ] )
    jet15c   = array( 'f', [ 0 ] )
    jet16pT  = array( 'f', [ 0 ] )#
    jet16eta = array( 'f', [ 0 ] )
    jet16phi = array( 'f', [ 0 ] )
    jet16b   = array( 'f', [ 0 ] )
    jet16c   = array( 'f', [ 0 ] )
    jet17pT  = array( 'f', [ 0 ] )#
    jet17eta = array( 'f', [ 0 ] )
    jet17phi = array( 'f', [ 0 ] )
    jet17b   = array( 'f', [ 0 ] )
    jet17c   = array( 'f', [ 0 ] )
    jet18pT  = array( 'f', [ 0 ] )#
    jet18eta = array( 'f', [ 0 ] )
    jet18phi = array( 'f', [ 0 ] )
    jet18b   = array( 'f', [ 0 ] )
    jet18c   = array( 'f', [ 0 ] )
    jet19pT  = array( 'f', [ 0 ] )#
    jet19eta = array( 'f', [ 0 ] )
    jet19phi = array( 'f', [ 0 ] )
    jet19b   = array( 'f', [ 0 ] )
    jet19c   = array( 'f', [ 0 ] )
    jet20pT  = array( 'f', [ 0 ] )#
    jet20eta = array( 'f', [ 0 ] )
    jet20phi = array( 'f', [ 0 ] )
    jet20b   = array( 'f', [ 0 ] )
    jet20c   = array( 'f', [ 0 ] )
    jet21pT  = array( 'f', [ 0 ] )#
    jet21eta = array( 'f', [ 0 ] )
    jet21phi = array( 'f', [ 0 ] )
    jet21b   = array( 'f', [ 0 ] )
    jet21c   = array( 'f', [ 0 ] )


    br_numlep  = tree.Branch( 'numlep' , numlep , 'numlep/F'  )
    br_numjet  = tree.Branch( 'numjet' , numjet , 'numjet/F'  )
    br_lep1pT  = tree.Branch( 'lep1pT' , lep1pT , 'lep1pT/F'  )
    br_lep1eta = tree.Branch( 'lep1eta', lep1eta, 'lep1eta/F' )
    br_lep1phi = tree.Branch( 'lep1phi', lep1phi, 'lep1phi/F' )
    br_lep2pT  = tree.Branch( 'lep2pT' , lep2pT , 'lep2pT/F'  )
    br_lep2eta = tree.Branch( 'lep2eta', lep2eta, 'lep2eta/F' )
    br_lep2phi = tree.Branch( 'lep2phi', lep2phi, 'lep2phi/F' )
    br_lep3pT  = tree.Branch( 'lep3pT' , lep3pT , 'lep3pT/F'  )
    br_lep3eta = tree.Branch( 'lep3eta', lep3eta, 'lep3eta/F' )
    br_lep3phi = tree.Branch( 'lep3phi', lep3phi, 'lep3phi/F' )
    br_mt1     = tree.Branch( 'mt1'    , mt1    , 'mt1/F'     )
    br_mt2     = tree.Branch( 'mt2'    , mt2    , 'mt2/F'     )
    br_mt3     = tree.Branch( 'mt3'    , mt3    , 'mt3/F'     )
    br_dr1     = tree.Branch( 'dr1'    , dr1    , 'dr1/F'     )
    br_dr2     = tree.Branch( 'dr2'    , dr2    , 'dr2/F'     )
    br_dr3     = tree.Branch( 'dr3'    , dr3    , 'dr3/F'     )
    br_btag    = tree.Branch( 'btag'   , btag   , 'btag/F'    )
    br_cent    = tree.Branch( 'cent'   , cent   , 'cent/F'    )
    br_srap    = tree.Branch( 'srap'   , srap   , 'srap/F'    )
    br_m_bb    = tree.Branch( 'm_bb'   , m_bb   , 'm_bb/F'    )
    br_h_b     = tree.Branch( 'h_b'    , h_b    , 'h_b/F'     )
    br_chi     = tree.Branch( 'chi'    , chi    , 'chi/F'     )
    br_jet1pT  = tree.Branch( 'jet1pT' , jet1pT , 'jet1pT/F'  )
    br_jet1eta = tree.Branch( 'jet1eta', jet1eta, 'jet1eta/F' )
    br_jet1phi = tree.Branch( 'jet1phi', jet1phi, 'jet1phi/F' )
    br_jet1b   = tree.Branch( 'jet1b'  , jet1b  , 'jet1b/F'   )
    br_jet1c   = tree.Branch( 'jet1c'  , jet1c  , 'jet1c/F'   )
    br_jet2pT  = tree.Branch( 'jet2pT' , jet2pT , 'jet2pT/F'  )#
    br_jet2eta = tree.Branch( 'jet2eta', jet2eta, 'jet2eta/F' )
    br_jet2phi = tree.Branch( 'jet2phi', jet2phi, 'jet2phi/F' )
    br_jet2b   = tree.Branch( 'jet2b'  , jet2b  , 'jet2b/F'   )
    br_jet2c   = tree.Branch( 'jet2c'  , jet2c  , 'jet2c/F'   )
    br_jet3pT  = tree.Branch( 'jet3pT' , jet3pT , 'jet3pT/F'  )#
    br_jet3eta = tree.Branch( 'jet3eta', jet3eta, 'jet3eta/F' )
    br_jet3phi = tree.Branch( 'jet3phi', jet3phi, 'jet3phi/F' )
    br_jet3b   = tree.Branch( 'jet3b'  , jet3b  , 'jet3b/F'   )
    br_jet3c   = tree.Branch( 'jet3c'  , jet3c  , 'jet4c/F'   )
    br_jet4pT  = tree.Branch( 'jet4pT' , jet4pT , 'jet4pT/F'  )#
    br_jet4eta = tree.Branch( 'jet4eta', jet4eta, 'jet4eta/F' )
    br_jet4phi = tree.Branch( 'jet4phi', jet4phi, 'jet4phi/F' )
    br_jet4b   = tree.Branch( 'jet4b'  , jet4b  , 'jet4b/F'   )
    br_jet4c   = tree.Branch( 'jet4c'  , jet4c  , 'jet5c/F'   )
    br_jet5pT  = tree.Branch( 'jet5pT' , jet5pT , 'jet5pT/F'  )#
    br_jet5eta = tree.Branch( 'jet5eta', jet5eta, 'jet5eta/F' )
    br_jet5phi = tree.Branch( 'jet5phi', jet5phi, 'jet5phi/F' )
    br_jet5b   = tree.Branch( 'jet5b'  , jet5b  , 'jet5b/F'   )
    br_jet5c   = tree.Branch( 'jet5c'  , jet5c  , 'jet5c/F'   )
    br_jet6pT  = tree.Branch( 'jet6pT' , jet6pT , 'jet6pT/F'  )#
    br_jet6eta = tree.Branch( 'jet6eta', jet6eta, 'jet6eta/F' )
    br_jet6phi = tree.Branch( 'jet6phi', jet6phi, 'jet6phi/F' )
    br_jet6b   = tree.Branch( 'jet6b'  , jet6b  , 'jet6b/F'   )
    br_jet6c   = tree.Branch( 'jet6c'  , jet6c  , 'jet6c/F'   )
    br_jet7pT  = tree.Branch( 'jet7pT' , jet7pT , 'jet7pT/F'  )#
    br_jet7eta = tree.Branch( 'jet7eta', jet7eta, 'jet7eta/F' )
    br_jet7phi = tree.Branch( 'jet7phi', jet7phi, 'jet7phi/F' )
    br_jet7b   = tree.Branch( 'jet7b'  , jet7b  , 'jet7b/F'   )
    br_jet7c   = tree.Branch( 'jet7c'  , jet7c  , 'jet7c/F'   )
    br_jet8pT  = tree.Branch( 'jet8pT' , jet8pT , 'jet8pT/F'  )#
    br_jet8eta = tree.Branch( 'jet8eta', jet8eta, 'jet8eta/F' )
    br_jet8phi = tree.Branch( 'jet8phi', jet8phi, 'jet8phi/F' )
    br_jet8b   = tree.Branch( 'jet8b'  , jet8b  , 'jet8b/F'   )
    br_jet8c   = tree.Branch( 'jet8c'  , jet8c  , 'jet8c/F'   )
    br_jet9pT  = tree.Branch( 'jet9pT' , jet9pT , 'jet9pT/F'  )#
    br_jet9eta = tree.Branch( 'jet9eta', jet9eta, 'jet9eta/F' )
    br_jet9phi = tree.Branch( 'jet9phi', jet9phi, 'jet9phi/F' )
    br_jet9b   = tree.Branch( 'jet9b'  , jet9b  , 'jet9b/F'   )
    br_jet9c   = tree.Branch( 'jet9c'  , jet9c  , 'jet9c/F'   )
    br_jet10pT  = tree.Branch( 'jet10pT' , jet10pT , 'jet10pT/F'  )#
    br_jet10eta = tree.Branch( 'jet10eta', jet10eta, 'jet10eta/F' )
    br_jet10phi = tree.Branch( 'jet10phi', jet10phi, 'jet10phi/F' )
    br_jet10b   = tree.Branch( 'jet10b'  , jet10b  , 'jet10b/F'   )
    br_jet10c   = tree.Branch( 'jet10c'  , jet10c  , 'jet10c/F'   )
    br_jet11pT  = tree.Branch( 'jet11pT' , jet11pT , 'jet11pT/F'  )#
    br_jet11eta = tree.Branch( 'jet11eta', jet11eta, 'jet11eta/F' )
    br_jet11phi = tree.Branch( 'jet11phi', jet11phi, 'jet11phi/F' )
    br_jet11b   = tree.Branch( 'jet11b'  , jet11b  , 'jet11b/F'   )
    br_jet11c   = tree.Branch( 'jet11c'  , jet11c  , 'jet11c/F'   )
    br_jet12pT  = tree.Branch( 'jet12pT' , jet12pT , 'jet12pT/F'  )#
    br_jet12eta = tree.Branch( 'jet12eta', jet12eta, 'jet12eta/F' )
    br_jet12phi = tree.Branch( 'jet12phi', jet12phi, 'jet12phi/F' )
    br_jet12b   = tree.Branch( 'jet12b'  , jet12b  , 'jet12b/F'   )
    br_jet12c   = tree.Branch( 'jet12c'  , jet12c  , 'jet12c/F'   )
    br_jet13pT  = tree.Branch( 'jet13pT' , jet13pT , 'jet13pT/F'  )#
    br_jet13eta = tree.Branch( 'jet13eta', jet13eta, 'jet13eta/F' )
    br_jet13phi = tree.Branch( 'jet13phi', jet13phi, 'jet13phi/F' )
    br_jet13b   = tree.Branch( 'jet13b'  , jet13b  , 'jet13b/F'   )
    br_jet13c   = tree.Branch( 'jet13c'  , jet13c  , 'jet13c/F'   )
    br_jet14pT  = tree.Branch( 'jet14pT' , jet14pT , 'jet14pT/F'  )#
    br_jet14eta = tree.Branch( 'jet14eta', jet14eta, 'jet14eta/F' )
    br_jet14phi = tree.Branch( 'jet14phi', jet14phi, 'jet14phi/F' )
    br_jet14b   = tree.Branch( 'jet14b'  , jet14b  , 'jet14b/F'   )
    br_jet14c   = tree.Branch( 'jet14c'  , jet14c  , 'jet14c/F'   )
    br_jet15pT  = tree.Branch( 'jet15pT' , jet15pT , 'jet15pT/F'  )#
    br_jet15eta = tree.Branch( 'jet15eta', jet15eta, 'jet15eta/F' )
    br_jet15phi = tree.Branch( 'jet15phi', jet15phi, 'jet15phi/F' )
    br_jet15b   = tree.Branch( 'jet15b'  , jet15b  , 'jet15b/F'   )
    br_jet15c   = tree.Branch( 'jet15c'  , jet15c  , 'jet15c/F'   )
    br_jet16pT  = tree.Branch( 'jet16pT' , jet16pT , 'jet16pT/F'  )#
    br_jet16eta = tree.Branch( 'jet16eta', jet16eta, 'jet16eta/F' )
    br_jet16phi = tree.Branch( 'jet16phi', jet16phi, 'jet16phi/F' )
    br_jet16b   = tree.Branch( 'jet16b'  , jet16b  , 'jet16b/F'   )
    br_jet16c   = tree.Branch( 'jet16c'  , jet16c  , 'jet16c/F'   )
    br_jet17pT  = tree.Branch( 'jet17pT' , jet17pT , 'jet17pT/F'  )#
    br_jet17eta = tree.Branch( 'jet17eta', jet17eta, 'jet17eta/F' )
    br_jet17phi = tree.Branch( 'jet17phi', jet17phi, 'jet17phi/F' )
    br_jet17b   = tree.Branch( 'jet17b'  , jet17b  , 'jet17b/F'   )
    br_jet17c   = tree.Branch( 'jet17c'  , jet17c  , 'jet17c/F'   )
    br_jet18pT  = tree.Branch( 'jet18pT' , jet18pT , 'jet18pT/F'  )#
    br_jet18eta = tree.Branch( 'jet18eta', jet18eta, 'jet18eta/F' )
    br_jet18phi = tree.Branch( 'jet18phi', jet18phi, 'jet18phi/F' )
    br_jet18b   = tree.Branch( 'jet18b'  , jet18b  , 'jet18b/F'   )
    br_jet18c   = tree.Branch( 'jet18c'  , jet18c  , 'jet18c/F'   )
    br_jet19pT  = tree.Branch( 'jet19pT' , jet19pT , 'jet19pT/F'  )#
    br_jet19eta = tree.Branch( 'jet19eta', jet19eta, 'jet19eta/F' )
    br_jet19phi = tree.Branch( 'jet19phi', jet19phi, 'jet19phi/F' )
    br_jet19b   = tree.Branch( 'jet19b'  , jet19b  , 'jet19b/F'   )
    br_jet19c   = tree.Branch( 'jet19c'  , jet19c  , 'jet19c/F'   )
    br_jet20pT  = tree.Branch( 'jet20pT' , jet20pT , 'jet20pT/F'  )#
    br_jet20eta = tree.Branch( 'jet20eta', jet20eta, 'jet20eta/F' )
    br_jet20phi = tree.Branch( 'jet20phi', jet20phi, 'jet20phi/F' )
    br_jet20b   = tree.Branch( 'jet20b'  , jet20b  , 'jet20b/F'   )
    br_jet20c   = tree.Branch( 'jet20c'  , jet20c  , 'jet20c/F'   )
    br_jet21pT  = tree.Branch( 'jet21pT' , jet21pT , 'jet21pT/F'  )#
    br_jet21eta = tree.Branch( 'jet21eta', jet21eta, 'jet21eta/F' )
    br_jet21phi = tree.Branch( 'jet21phi', jet21phi, 'jet21phi/F' )
    br_jet21b   = tree.Branch( 'jet21b'  , jet21b  , 'jet21b/F'   )
    br_jet21c   = tree.Branch( 'jet21c'  , jet21c  , 'jet21c/F'   )


    # track the time
    start_time = time.clock()

    def missingPT(x):
        met_value = ROOT.TMath.Sqrt(2 * event.met[0] * event.leppT[x]/(10**6) * ( 1 - ROOT.TMath.Cos((lepvec[x].DeltaPhi(neutrino[0])))))
        return met_value
    # Average separation in pseudorapidity between two b-tagged jets.
    def etabi_j(x,y):
        distance = abs(jetvec[tracker_btj[x]].Eta() - jetvec[tracker_btj[y]].Eta())
        return distance
    # Vector Pt or M Sum between two b-tagged jets.
    def vectorsum(x,y,c):
        if c == 'Pt':
            sum = (jetvec[tracker_btj[x]] + jetvec[tracker_btj[y]]).Pt()
        elif c == 'M':
            sum = (jetvec[tracker_btj[x]] + jetvec[tracker_btj[y]]).M()
        return sum
#############################################################################

    n_entries = tree.GetEntries()
    i= 1
    for event in tree:
        #####################################################################
        tracker_btj = []            # Initialize empty tracking btagjets.
        tracker_non = []            # Initialize empty tracking lightjets.
        m1          = 0.
        m2          = 0.
        mv          = 0.
        l1          = []
        l2          = []
        l3          = []
        dR1         = []
        dR2         = []
        dR3         = []
        chisq       = []
        lepvec      = {}
        jetvec      = {}
        neutrino    = {}
        HB_sum_Pt   = 0.             # Initialize sum of Pt for all b-tag jets.
        rand        = 0.
        cen_sum_Pt  = 0.             # Initialize sum of Pt for all jets.
        cen_sum_E   = 0.             # Initialize sum of E for all jets.
        etasum      = 0.             # Initialize sum for eta seperation.
        etasum_N    = 0.             # Initialize sum for eta separation average.
        btjmaxPt    = 0.             # Initialize empty b-tag vecto for max .Pt().
        btjmaxM     = 0.             # Initialize empty b-tag vecto for max .M().
        vec_sum_Pt  = 0.            # Initialize empty b-tag vector for summing Pt().
        vec_sum_M   = 0.            # Initialize empty b-tag vector for summing M().
        ####################################################################
        # show some progress
        if i % 1000 == 0: print("   processing entry {:8d}/{:d} [{:5.0f} evts/s]".format(i, n_entries, i/(time.clock()-start_time)))
        numlep[0] = lep = event.nlep[0]
        numjet[0] = jet = event.njet[0]
        if lep > 0:  
            neutrino[0] = ROOT.TLorentzVector()
            neutrino[0].SetPtEtaPhiM(event.met[0],0,event.met_phi[0],0)
            for j in xrange(lep):
                lepvec[j] = ROOT.TLorentzVector()
                lepvec[j].SetPtEtaPhiM(event.leppT[j],event.lepeta[j],event.lepphi[j],0)
            for k in xrange(jet):
                jetvec[k] = ROOT.TLorentzVector()  
                jetvec[k].SetPtEtaPhiM(event.jetpT[k],event.jeteta[k],event.jetphi[k],0)
                if lep > 0:
                    dR1.append(lepvec[0].DeltaR(jetvec[k]))
                if lep > 1: 
                    dR2.append(lepvec[1].DeltaR(jetvec[k]))
                if lep > 2:
                    dR3.append(lepvec[2].DeltaR(jetvec[k]))
            for x in xrange(jet):
                cen_sum_E  += jetvec[x].E()          # Scalar sum of E.
                cen_sum_Pt += jetvec[x].Pt()         # Scalar sum of Pt.
                rand = random.random()

                if event.jetbhadron[x] == 1 and rand <= 0.7:
                    tracker_btj.append(x)              # B-tag jets into a list.
                elif event.jetchadron[x] == 1 and rand <= 0.2:
                    tracker_btj.append(x)
                elif rand <= 0.002:
                    tracker_btj.append(x)
                else:
                    tracker_non.append(x)
            btagjets = len(tracker_btj)
            ntagjets = len(tracker_non)
            btag[0]  = btagjets 
            if cen_sum_E != 0:
                cent[0] = cen_sum_Pt / cen_sum_E    # scalar sum of Pt/E.
            else:
                cent[0] = -9999
            for k in xrange(btagjets):
                HB_sum_Pt += jetvec[tracker_btj[k]].Pt()
                for j in xrange(btagjets):
                    if k == j: continue
                    etasum += etabi_j(k,j)           # Finding separation between all b_jets.
                    vec_sum_Pt = vectorsum(k,j,'Pt') # Sum of btagjets Pt.
                    vec_sum_M  = vectorsum(k,j,'M')  # Sum of btagjets M.
                    if vec_sum_Pt < btjmaxPt:continue
                    # Finds max Pt and M for two btagjets.
                    btjmaxPt = vec_sum_Pt
                    btjmaxM  = vec_sum_M
            m_bb[0] = btjmaxM/1000
            h_b[0]  = HB_sum_Pt/1000
            if btagjets > 1:
                etasum_N = etasum/(btagjets**2 - btagjets)  # Getting distance avg.
            else:
                etasum_N = -999

            srap[0] = etasum_N                        # btagjets speration avg.
            if btagjets >= 6:
                for o in xrange(0,6):
                    for j in xrange(1,6):
                        for k in xrange(2,6):
                            for l in xrange(3,6):
                                if o == j == k == l :continue
                                if o > j > k > l :continue
                                if j > k > l :continue
                                if k > l :continue
                                if o > j :continue
                                if j > k  :continue
                                if o == j == k :continue
                                if o == j :continue
                                if o == k :continue
                                if o == l :continue
                                if j == k == l:continue
                                if j == k :continue
                                if j == l :continue
                                if k == l :continue
                                chisq.append((vectorsum(o,j,'M') - 120000)**2 + (vectorsum(k,l,'M') - 120000)**2)
            if len(chisq) > 0:
                chi[0] = min(chisq)
            else:
                chi[0] = -999
    ######################################################################################
        if lep >0:
            lep1pT[0]  = event.leppT[0]
            lep1eta[0] = event.lepeta[0]
            lep1phi[0] = event.lepphi[0]
            mt1[0]     = missingPT(0)
            if len(dR1) > 0 : dr1[0]  = min(dR1)
            if lep > 1:
                lep2pT[0]  = event.leppT[1]
                lep2eta[0] = event.lepeta[1]
                lep2phi[0] = event.lepphi[1]
                mt2[0] = missingPT(1)
                if len(dR2) > 0 : dr2[0]  = min(dR2)
                if lep > 2:
                    lep3pT[0]  = event.leppT[2]
                    lep3eta[0] = event.lepeta[2]
                    lep3phi[0] = event.lepphi[2]
                    mt3[0] = missingPT(2)
                    if len(dR3) > 0 : dr3[0]  = min(dR3)
                else:
                    lep3pT[0]  = -999
                    lep3eta[0] = -9
                    lep3phi[0] = -9
                    mt3[0]     = -999
                    dr3[0]     = -999
            else:
                lep2pT[0]  = -999
                lep2eta[0] = -9
                lep2phi[0] = -9
                mt2[0]     = -999
                dr2[0]     = -999
        else:
            lep1pT[0]  = -999
            lep1eta[0] = -9
            lep1phi[0] = -9
            mt1[0]     = -999
            dr1[0]     = -999
        # event.jetpT[k],event.jeteta[k],event.jetphi[k]
        if jet > 0: 
            jet1pT  = event.jetpT[0]
            jet1eta = event.jeteta[0]
            jet1phi = event.jetphi[0]
            jet1b   = event.jetbhadron[0]
            jet1c   = event.jetchadron[0]
            if jet >1:
                jet2pT  = event.jetpT[1]
                jet2eta = event.jeteta[1]
                jet2phi = event.jetphi[1]
                jet2b   = event.jetbhadron[1]
                jet2c   = event.jetchadron[1]
                if jet > 2:
                    jet3pT  = event.jetpT[2]
                    jet3eta = event.jeteta[2]
                    jet3phi = event.jetphi[2]
                    jet3b   = event.jetbhadron[2]
                    jet3c   = event.jetchadron[2]
                    if jet > 3 
                        jet4pT  = event.jetpT[3]
                        jet4eta = event.jeteta[3]
                        jet4phi = event.jetphi[3]
                        jet4b   = event.jetbhadron[3]
                        jet4c   = event.jetchadron[3]
                        if jet >4:
                            jet5pT  = event.jetpT[4]
                            jet5eta = event.jeteta[4]
                            jet5phi = event.jetphi[4]
                            jet5b   = event.jetbhadron[4]
                            jet5c   = event.jetchadron[4]
                            if jet > 5:
                                jet6pT  = event.jetpT[5]
                                jet6eta = event.jeteta[5]
                                jet6phi = event.jetphi[5]
                                jet6b   = event.jetbhadron[5]
                                jet6c   = event.jetchadron[5]
                                if jet > 6: 
                                    jet7pT  = event.jetpT[6]
                                    jet7eta = event.jeteta[6]
                                    jet7phi = event.jetphi[6]
                                    jet7b   = event.jetbhadron[6]
                                    jet7c   = event.jetchadron[6]
                                    if jet >7:
                                        jet8pT  = event.jetpT[7]
                                        jet8eta = event.jeteta[7]
                                        jet8phi = event.jetphi[7]
                                        jet8b   = event.jetbhadron[7]
                                        jet8c   = event.jetchadron[7]
                                        if jet > 8:
                                            jet9pT  = event.jetpT[8]
                                            jet9eta = event.jeteta[8]
                                            jet9phi = event.jetphi[8]
                                            jet9b   = event.jetbhadron[8]
                                            jet9c   = event.jetchadron[8]
                                            if jet > 9 
                                                jet10pT  = event.jetpT[9]
                                                jet10eta = event.jeteta[9]
                                                jet10phi = event.jetphi[9]
                                                jet10b   = event.jetbhadron[9]
                                                jet10c   = event.jetchadron[9]
                                                if jet > 10:
                                                    jet11pT  = event.jetpT[10]
                                                    jet11eta = event.jeteta[10]
                                                    jet11phi = event.jetphi[10]
                                                    jet11b   = event.jetbhadron[10]
                                                    jet11c   = event.jetchadron[10]
                                                    if jet > 11:
                                                        jet12pT  = event.jetpT[11]
                                                        jet12eta = event.jeteta[11]
                                                        jet12phi = event.jetphi[11]
                                                        jet12b   = event.jetbhadron[11]
                                                        jet12c   = event.jetchadron[11]
                                                        if jet >12:
                                                            jet13pT  = event.jetpT[12]
                                                            jet13eta = event.jeteta[12]
                                                            jet13phi = event.jetphi[12]
                                                            jet13b   = event.jetbhadron[12]
                                                            jet13c   = event.jetchadron[12]
                                                            if jet > 13:
                                                                jet14pT  = event.jetpT[13]
                                                                jet14eta = event.jeteta[13]
                                                                jet14phi = event.jetphi[13]
                                                                jet14b   = event.jetbhadron[13]
                                                                jet14c   = event.jetchadron[13]
                                                                if jet > 14:
                                                                    jet15pT  = event.jetpT[14]
                                                                    jet15eta = event.jeteta[14]
                                                                    jet15phi = event.jetphi[14]
                                                                    jet15b   = event.jetbhadron[14]
                                                                    jet15c   = event.jetchadron[14]
                                                                    if jet > 15:
                                                                        jet16pT  = event.jetpT[15]
                                                                        jet16eta = event.jeteta[15]
                                                                        jet16phi = event.jetphi[15]
                                                                        jet16b   = event.jetbhadron[15]
                                                                        jet16c   = event.jetchadron[15]
                                                                        if jet >16:
                                                                            jet17pT  = event.jetpT[16]
                                                                            jet17eta = event.jeteta[16]
                                                                            jet17phi = event.jetphi[16]
                                                                            jet17b   = event.jetbhadron[16]
                                                                            jet17c   = event.jetchadron[16]
                                                                            if jet > 17:
                                                                                jet18pT  = event.jetpT[17]
                                                                                jet18eta = event.jeteta[17]
                                                                                jet18phi = event.jetphi[17]
                                                                                jet18b   = event.jetbhadron[17]
                                                                                jet18c   = event.jetchadron[17]
                                                                                if jet > 18:
                                                                                    jet19pT  = event.jetpT[18]
                                                                                    jet19eta = event.jeteta[18]
                                                                                    jet19phi = event.jetphi[18]
                                                                                    jet19b   = event.jetbhadron[18]
                                                                                    jet19c   = event.jetchadron[18]
                                                                                    if jet > 19:
                                                                                        jet20pT  = event.jetpT[19]
                                                                                        jet20eta = event.jeteta[19]
                                                                                        jet20phi = event.jetphi[19]
                                                                                        jet20b   = event.jetbhadron[19]
                                                                                        jet20c   = event.jetchadron[19]
                                                                                        if jet > 20:
                                                                                            jet21pT  = event.jetpT[20]
                                                                                            jet21eta = event.jeteta[20]
                                                                                            jet21phi = event.jetphi[20]
                                                                                            jet21b   = event.jetbhadron[20]
                                                                                            jet21c   = event.jetchadron[20]
                                                                                        else:
                                                                                            jet21pT  = -999
                                                                                            jet21eta = -9
                                                                                            jet21phi = -9
                                                                                            jet21b   = -9
                                                                                            jet21c   = -9
                                                                                    else:
                                                                                        jet20pT  = -999
                                                                                        jet20eta = -9
                                                                                        jet20phi = -9
                                                                                        jet20b   = -9
                                                                                        jet20c   = -9
                                                                                else:
                                                                                    jet19pT  = -999
                                                                                    jet19eta = -9
                                                                                    jet19phi = -9
                                                                                    jet19b   = -9
                                                                                    jet19c   = -9
                                                                            else:
                                                                                jet18pT  = -999
                                                                                jet18eta = -9
                                                                                jet18phi = -9
                                                                                jet18b   = -9
                                                                                jet18c   = -9
                                                                        else:
                                                                            jet17pT  = -999
                                                                            jet17eta = -9
                                                                            jet17phi = -9
                                                                            jet17b   = -9
                                                                            jet17c   = -9
                                                                    else:
                                                                        jet16pT  = -999
                                                                        jet16eta = -9
                                                                        jet16phi = -9
                                                                        jet16b   = -9
                                                                        jet16c   = -9
                                                                else:
                                                                    jet15pT  = -999
                                                                    jet15eta = -9
                                                                    jet15phi = -9
                                                                    jet15b   = -9
                                                                    jet15c   = -9
                                                            else:
                                                                jet14pT  = -999
                                                                jet14eta = -9
                                                                jet14phi = -9
                                                                jet14b   = -9
                                                                jet14c   = -9
                                                        else:
                                                            jet13pT  = -999
                                                            jet13eta = -9
                                                            jet13phi = -9
                                                            jet13b   = -9
                                                            jet13c   = -9
                                                    else:
                                                        jet12pT  = -999
                                                        jet12eta = -9
                                                        jet12phi = -9
                                                        jet12b   = -9
                                                        jet12c   = -9
                                                else:
                                                    jet11pT  = -999
                                                    jet11eta = -9
                                                    jet11phi = -9
                                                    jet11b   = -9
                                                    jet11c   = -9
                                            else:
                                                jet10pT  = -999
                                                jet10eta = -9
                                                jet10phi = -9
                                                jet10b   = -9
                                                jet10c   = -9
                                        else:
                                            jet9pT  = -999
                                            jet9eta = -9
                                            jet9phi = -9
                                            jet9b   = -9
                                            jet9c   = -9
                                    else:
                                        jet8pT  = -999
                                        jet8eta = -9
                                        jet8phi = -9
                                        jet8b   = -9
                                        jet8c   = -9
                                else:
                                    jet7pT  = -999
                                    jet7eta = -9
                                    jet7phi = -9
                                    jet7b   = -9
                                    jet7c   = -9
                            else:
                                jet6pT  = -999
                                jet6eta = -9
                                jet6phi = -9
                                jet6b   = -9
                                jet6c   = -9
                        else:
                            jet5pT  = -999
                            jet5eta = -9
                            jet5phi = -9
                            jet5b   = -9
                            jet5c   = -9
                    else:
                        jet4pT  = -999
                        jet4eta = -9
                        jet4phi = -9
                        jet4b   = -9
                        jet4c   = -9
                else:
                    jet3pT  = -999
                    jet3eta = -9
                    jet3phi = -9
                    jet3b   = -9
                    jet3c   = -9
            else:
                jet2pT  = -999
                jet2eta = -9
                jet2phi = -9
                jet2b   = -9
                jet2c   = -9
        else:
            jet1pT  = -999
            jet1eta = -9
            jet1phi = -9
            jet1b   = -9
            jet1c   = -9


        # fill new branches
        br_numjet.Fill()
        br_numlep.Fill()
        br_lep1pT.Fill()
        br_lep1eta.Fill()
        br_lep1phi.Fill()
        br_lep2pT.Fill()
        br_lep2eta.Fill()
        br_lep2phi.Fill()
        br_lep3pT.Fill()
        br_lep3eta.Fill()
        br_lep3phi.Fill()
        br_mt1.Fill()
        br_mt2.Fill()
        br_mt3.Fill()
        br_dr1.Fill()
        br_dr2.Fill()
        br_dr3.Fill()
        br_btag.Fill()
        br_cent.Fill() 
        br_srap.Fill()
        br_m_bb.Fill()
        br_h_b.Fill()
        br_chi.Fill()
        br_jet1pT.Fill()
        br_jet1eta.Fill()
        br_jet1phi.Fill()
        br_jet1b.Fill()
        br_jet1c.Fill()
        br_jet2pT.Fill()
        br_jet2eta.Fill()
        br_jet2phi.Fill()
        br_jet2b.Fill()
        br_jet2c.Fill()
        br_jet3pT.Fill()
        br_jet3eta.Fill()
        br_jet3phi.Fill()
        br_jet3b.Fill()
        br_jet3c.Fill()
        br_jet4pT.Fill()
        br_jet4eta.Fill()
        br_jet4phi.Fill()
        br_jet4b.Fill()
        br_jet4c.Fill()
        br_jet5pT.Fill()
        br_jet5eta.Fill()
        br_jet5phi.Fill()
        br_jet5b.Fill()
        br_jet5c.Fill()
        br_jet6pT.Fill()
        br_jet6eta.Fill()
        br_jet6phi.Fill()
        br_jet6b.Fill()
        br_jet6c.Fill()
        br_jet7pT.Fill()
        br_jet7eta.Fill()
        br_jet7phi.Fill()
        br_jet7b.Fill()
        br_jet7c.Fill()
        br_jet8pT.Fill()
        br_jet8eta.Fill()
        br_jet8phi.Fill()
        br_jet8b.Fill()
        br_jet8c.Fill()
        br_jet9pT.Fill()
        br_jet9eta.Fill()
        br_jet9phi.Fill()
        br_jet9b.Fill()
        br_jet9c.Fill()
        br_jet10pT.Fill()
        br_jet10eta.Fill()
        br_jet10phi.Fill()
        br_jet10b.Fill()
        br_jet10c.Fill()
        br_jet11pT.Fill()
        br_jet11eta.Fill()
        br_jet11phi.Fill()
        br_jet11b.Fill()
        br_jet11c.Fill()
        br_jet12pT.Fill()
        br_jet12eta.Fill()
        br_jet12phi.Fill()
        br_jet12b.Fill()
        br_jet12c.Fill()
        br_jet13pT.Fill()
        br_jet13eta.Fill()
        br_jet13phi.Fill()
        br_jet13b.Fill()
        br_jet13c.Fill()
        br_jet14pT.Fill()
        br_jet14eta.Fill()
        br_jet14phi.Fill()
        br_jet14b.Fill()
        br_jet14c.Fill()
        br_jet15pT.Fill()
        br_jet15eta.Fill()
        br_jet15phi.Fill()
        br_jet15b.Fill()
        br_jet15c.Fill()
        br_jet16pT.Fill()
        br_jet16eta.Fill()
        br_jet16phi.Fill()
        br_jet16b.Fill()
        br_jet16c.Fill()
        br_jet17pT.Fill()
        br_jet17eta.Fill()
        br_jet17phi.Fill()
        br_jet17b.Fill()
        br_jet17c.Fill()
        br_jet18pT.Fill()
        br_jet18eta.Fill()
        br_jet18phi.Fill()
        br_jet18b.Fill()
        br_jet18c.Fill()
        br_jet19pT.Fill()
        br_jet19eta.Fill()
        br_jet19phi.Fill()
        br_jet19b.Fill()
        br_jet19c.Fill()
        br_jet20pT.Fill()
        br_jet20eta.Fill()
        br_jet20phi.Fill()
        br_jet20b.Fill()
        br_jet20c.Fill()
        br_jet21pT.Fill()
        br_jet21eta.Fill()
        br_jet21phi.Fill()
        br_jet21b.Fill()
        br_jet21c.Fill()
        i += 1

    # write augmented tree to original file
    tree.Write("", ROOT.TObject.kOverwrite)

    t.Close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='flatten/augment ntuples')
    parser.add_argument('--file', help='input file to skim')
    args = parser.parse_args()

    augment_rootfile(args.file)
