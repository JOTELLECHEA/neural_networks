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
        jet1pT  = event.jetpT[0]
        jet1eta = event.jeteta[0]
        jet1phi = event.jetphi[0]
        jet1b   = event.jetbhadron[0]
        jet1c   = event.jetchadron[0]


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
        i += 1

    # write augmented tree to original file
    tree.Write("", ROOT.TObject.kOverwrite)

    t.Close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='flatten/augment ntuples')
    parser.add_argument('--file', help='input file to skim')
    args = parser.parse_args()

    augment_rootfile(args.file)
