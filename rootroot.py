import ROOT

treename = "OutputTree"

file = "background.root"
t = ROOT.TFile(file)
tree = t.Get(treename)

# h1  = ROOT.TH1D('jet','jet;Jet muliplicity;Events normalised to unit area',21,0,21)
# h2  = ROOT.TH1D('lep','lep;lep muliplicity;Events normalised to unit area',3,0,3)
j1 = ROOT.TH1D("jet1pT", "jet1pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j2 = ROOT.TH1D("jet2pT", "jet2pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j3 = ROOT.TH1D("jet3pT", "jet3pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j4 = ROOT.TH1D("jet4pT", "jet4pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j5 = ROOT.TH1D("jet5pT", "jet5pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j6 = ROOT.TH1D("jet6pT", "jet6pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j7 = ROOT.TH1D("jet7pT", "jet7pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j8 = ROOT.TH1D("jet8pT", "jet8pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j9 = ROOT.TH1D("jet9pT", "jet9pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800)
j10 = ROOT.TH1D(
    "jet10pT", "jet10pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800
)
j11 = ROOT.TH1D(
    "jet11pT", "jet11pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800
)
j12 = ROOT.TH1D(
    "jet12pT", "jet12pT;Jet1 pT;Events normalised to unit area", 1800, 0, 1800
)
j13 = ROOT.TH1D(
    "jet13pT", "jet13pT;Jet1 pT;Events normalised to unit area", 1000, 0, 1000
)
j14 = ROOT.TH1D(
    "jet14pT", "jet14pT;Jet1 pT;Events normalised to unit area", 1000, 0, 1000
)
j15 = ROOT.TH1D(
    "jet15pT", "jet15pT;Jet1 pT;Events normalised to unit area", 1000, 0, 1000
)
j16 = ROOT.TH1D(
    "jet16pT", "jet16pT;Jet1 pT;Events normalised to unit area", 800, 0, 800
)
j17 = ROOT.TH1D(
    "jet17pT", "jet17pT;Jet1 pT;Events normalised to unit area", 800, 0, 800
)
j18 = ROOT.TH1D(
    "jet18pT", "jet18pT;Jet1 pT;Events normalised to unit area", 800, 0, 800
)
j19 = ROOT.TH1D(
    "jet19pT", "jet19pT;Jet1 pT;Events normalised to unit area", 200, 0, 200
)
j20 = ROOT.TH1D(
    "jet20pT", "jet20pT;Jet1 pT;Events normalised to unit area", 200, 0, 200
)
j21 = ROOT.TH1D(
    "jet21pT", "jet21pT;Jet1 pT;Events normalised to unit area", 200, 0, 200
)
entries = tree.GetEntries()

for event in tree:
    w = event.mcweight[0]
    j1.Fill(event.jet1pT / 1000, w)
    j2.Fill(event.jet2pT / 1000, w)
    j3.Fill(event.jet3pT / 1000, w)
    j4.Fill(event.jet4pT / 1000, w)
    j5.Fill(event.jet5pT / 1000, w)
    j6.Fill(event.jet6pT / 1000, w)
    j7.Fill(event.jet7pT / 1000, w)
    j8.Fill(event.jet8pT / 1000, w)
    j9.Fill(event.jet9pT / 1000, w)
    j10.Fill(event.jet10pT / 1000, w)
    j11.Fill(event.jet11pT / 1000, w)
    j12.Fill(event.jet12pT / 1000, w)
    j13.Fill(event.jet13pT / 1000, w)
    j14.Fill(event.jet14pT / 1000, w)
    j15.Fill(event.jet15pT / 1000, w)
    j16.Fill(event.jet16pT / 1000, w)
    j17.Fill(event.jet17pT / 1000, w)
    j18.Fill(event.jet18pT / 1000, w)
    j19.Fill(event.jet19pT / 1000, w)
    j20.Fill(event.jet20pT / 1000, w)
    j21.Fill(event.jet21pT / 1000, w)
print "Done"


def plot(hist):
    hist.SetStats(0)
    hist.Scale(1 / (hist.Integral()))
    hist.Draw("HIST")


c1 = ROOT.TCanvas("c1", "c1", 200, 10, 700, 500)
c1.Divide(3, 2, 0.01, 0.01, 0)
c2 = ROOT.TCanvas("c2", "c2", 200, 10, 700, 500)
c2.Divide(3, 2, 0.01, 0.01, 0)
c3 = ROOT.TCanvas("c3", "c3", 200, 10, 700, 500)
c3.Divide(3, 2, 0.01, 0.01, 0)
c4 = ROOT.TCanvas("c4", "c4", 200, 10, 700, 500)
c4.Divide(3, 2, 0.01, 0.01, 0)

c1.cd(1)
plot(j1)
c1.cd(2)
plot(j2)
c1.cd(3)
plot(j3)
c1.cd(4)
plot(j4)
c1.cd(5)
plot(j5)
c1.cd(6)
plot(j6)

c2.cd(1)
plot(j7)
c2.cd(2)
plot(j8)
c2.cd(3)
plot(j9)
c2.cd(4)
plot(j10)
c2.cd(5)
plot(j11)
c2.cd(6)
plot(j12)

c3.cd(1)
plot(j13)
c3.cd(2)
plot(j14)
c3.cd(3)
plot(j15)
c3.cd(4)
plot(j16)
c3.cd(5)
plot(j17)
c3.cd(6)
plot(j18)

c4.cd(1)
plot(j19)
c4.cd(2)
plot(j20)
c4.cd(3)
plot(j21)
