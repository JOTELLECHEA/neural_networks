#!/usr/bin/env python2

import os, time, sys, argparse,math

import numpy as np
from array import array
import shutil
import random
import itertools


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
    btag    = array( 'f', [ 0 ] )
    srap    = array( 'f', [ 0 ] )
    cent    = array( 'f', [ 0 ] )
    m_bb    = array( 'f', [ 0 ] )
    h_b     = array( 'f', [ 0 ] )
    chi     = array( 'f', [ 0 ] )
    leptonpT={}
    leptoneta={}
    leptonphi={}
    mt={}
    dr={}
    maxlepton=4
    for i in range(1,maxlepton):
        leptonpT[i]  = array( 'f', [ 0 ] )
        leptoneta[i] = array( 'f', [ 0 ] )
        leptonphi[i] = array( 'f', [ 0 ] )
        mt[i]        = array( 'f', [ 0 ] )
        dr[i]        = array( 'f', [ 0 ] )
    br_leptonpT={}
    br_leptoneta={}
    br_leptonphi={}
    br_mt={}
    br_dr={}
    for i in range(1,maxlepton):
        br_leptonpT[i]  = tree.Branch('lepton%dpT'%i  , leptonpT[i] , 'lepton%dpT/F'%i)
        br_leptoneta[i] = tree.Branch('lepton%deta'%i , leptoneta[i] , 'lepton%deta/F'%i)
        br_leptonphi[i] = tree.Branch('lepton%dphi'%i , leptonphi[i] , 'lepton%dphi/F'%i)
        br_mt[i]        = tree.Branch('mt%d'%i , mt[i] , 'mt%d/F'%i)
        br_dr[i]        = tree.Branch('dr%d'%i , dr[i] , 'dr%d/F'%i)
    jetpT={}
    jeteta={}
    jetphi={}
    jetb={}
    jetc={}
    maxjets=22
    for i in range(1,maxjets):
        jetpT[i]  = array( 'f', [ 0 ] )
        jeteta[i] = array( 'f', [ 0 ] )
        jetphi[i] = array( 'f', [ 0 ] )
        jetb[i]   = array( 'f', [ 0 ] )
        jetc[i]   = array( 'f', [ 0 ] )
    br_jetpT={}
    br_jeteta={}
    br_jetphi={}
    br_jetc={}
    br_jetb={}
    for i in range(1,maxjets):
        br_jetpT[i]  = tree.Branch('jet%dpT'%i , jetpT[i] , 'jet%dpT/F'%i)
        br_jeteta[i] = tree.Branch('jet%deta'%i , jeteta[i] , 'jet%deta/F'%i)
        br_jetphi[i] = tree.Branch('jet%dphi'%i , jetphi[i] , 'jet%dphi/F'%i)
        br_jetb[i]   = tree.Branch('jet%db'%i , jetb[i] , 'jet%db/F'%i)
        br_jetc[i]   = tree.Branch('jet%dc'%i , jetc[i] , 'jet%dc/F'%i)
    br_numlep  = tree.Branch( 'numlep' , numlep , 'numlep/F'  )
    br_numjet  = tree.Branch( 'numjet' , numjet , 'numjet/F'  )
    br_btag    = tree.Branch( 'btag'   , btag   , 'btag/F'    )
    br_cent    = tree.Branch( 'cent'   , cent   , 'cent/F'    )
    br_srap    = tree.Branch( 'srap'   , srap   , 'srap/F'    )
    br_m_bb    = tree.Branch( 'm_bb'   , m_bb   , 'm_bb/F'    )
    br_h_b     = tree.Branch( 'h_b'    , h_b    , 'h_b/F'     )
    br_chi     = tree.Branch( 'chi'    , chi    , 'chi/F'     )



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
        delta_r = [dR1,dR2,dR3]
        lloop=range(1,maxlepton)
        for n in lloop:
            if n <= lep:
                leptonpT[n]  = event.leppT[n-1]
                leptoneta[n] = event.lepeta[n-1]
                leptonphi[n] = event.lepphi[n-1]
                mt[n]        = missingPT(n-1)
                dr[n] = min(delta_r[n-1])
                if len(delta_r[n-1]) == 0: 
                    dr[n] = -999
                else:
                    dr[n] = min(delta_r[n-1])
            else:
                leptonpT[n]  = -999
                leptoneta[n] = -9
                leptonphi[n] = -9
                mt[n]        = -999
                dr[n] = -999
        for n in lloop:
            br_leptonpT[n].Fill()
            br_leptoneta[n].Fill()
            br_leptonphi[n].Fill()
            br_mt[n].Fill()
            br_dr[n].Fill()

        jloop=range(1,maxjets)
        for n in jloop:
            if n <= jet:
                jetpT[n] = event.jetpT[n-1]
                jeteta[n] = event.jeteta[n-1]
                jetphi[n] = event.jetphi[n-1]
                jetb[n] = event.jetbhadron[n-1]
                jetc[n] = event.jetchadron[n-1]
            else:
                jetpT[n]  = -999
                jeteta[n] = -9
                jetphi[n] = -9
                jetb[n]   = -9
                jetc[n]   = -9
        for n in jloop:
            br_jetpT[n].Fill()
            br_jeteta[n].Fill()
            br_jetphi[n].Fill()
            br_jetb[n].Fill()
            br_jetc[n].Fill() 

        # fill new branches
        br_numjet.Fill()
        br_numlep.Fill()
        br_btag.Fill()
        br_cent.Fill() 
        br_srap.Fill()
        br_m_bb.Fill()
        br_h_b.Fill()
        br_chi.Fill()
      
        i += 1

    # write augmented tree to original file
    tree.Write("", ROOT.TObject.kOverwrite)

    t.Close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='flatten/augment ntuples')
    parser.add_argument('--file', help='input file to skim')
    args = parser.parse_args()

    augment_rootfile(args.file)
