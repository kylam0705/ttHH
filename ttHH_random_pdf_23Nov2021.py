import pandas
import awkward
import numpy
import matplotlib.pyplot as plt
from yahist import Hist1D
#import argparse

#def parse_arguments():
#	parser =argparse.ArgumentParser()
#	parser.add_argument(
#		"--sideband_cut",
#		required = False,
#		default = None,
#		type = str,
#		help = "places a cut to establish sideband definition")

events_awkward = awkward.from_parquet("/home/users/smay/public_html/forKyla/merged_nominal.parquet")
events = awkward.to_pandas(events_awkward)

#Data
events_data = events[events["process_id"] == 14]
events_data["MinPhoton_mvaID"] = events_data[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].min(axis=1)
sideband_cut = events_data[events_data["MinPhoton_mvaID"] < 0.0]

#Gamma + Jets Process:
events_GJets = events[(events["process_id"] >= 15) & (events["process_id"] <=19)]
#Prompt Photons
prompt_lead = events_GJets[events_GJets["LeadPhoton_genPartFlav"] == 1]
prompt_lead_id = prompt_lead["LeadPhoton_mvaID"]
prompt_sublead = events_GJets[events_GJets["SubleadPhoton_genPartFlav"] == 1]
prompt_sublead_id = prompt_sublead["SubleadPhoton_mvaID"]
#Prompt ID
prompt_id = pandas.concat([prompt_lead_id, prompt_sublead_id])

#Fake Photons
fake_lead = events_GJets[events_GJets["LeadPhoton_genPartFlav"] == 0]
fake_lead_id = fake_lead["LeadPhoton_mvaID"]
fake_sublead = events_GJets[events_GJets["SubleadPhoton_genPartFlav"] == 0]
fake_sublead_id = fake_sublead["SubleadPhoton_mvaID"]
#Fake ID
fake_id = pandas.concat([fake_lead_id, fake_sublead_id]) #Creates an array of fake photons to be inserted into the histogram h_fake

#More Columns in the Dataframe
#min_value_series = events['LeadPhoton_mvaID','SubleadPhoton_mvaID'].min(axis=1)
#events["MinPhoton_mvaID"] = min_value_series
#events_photon_sideband_cut = events[events["MinPhoton_mvaID"] > -0.67]
#events_photon_preselection_cut = events[events["max_pho_idmva"] > -0.67]

#Making the PDF Histogram
f = plt.figure()
h_fake = Hist1D(fake_id, bins = "40,-1,1") #Histogram of fake photons from fake_id array
h_fake = h_fake.normalize()
p = h_fake.counts #p-value in the random.choice function
#print(p)

#PDF Function

fake_photons_pdf = numpy.random.choice(a=14, size=sideband_cut.size, p=h_fake.counts[6:20])
 
h_pdf = Hist1D(fake_photons_pdf, bins = "40,-1,1") #This is the pdf that needs to be plotted in a histogram
h_pdf.plot(histtype="stepfilled", alpha = 0.8)

h_fake.plot(histtype="stepfilled", alpha = 0.8)
plt.yscale("log")
plt.title("Fake Photon IDMVA in GJets")
plt.xlabel("IDMVA Score")
plt.ylabel("Events")
#plt.legend()
plt.show()
#f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_GJets.pdf")
f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_mvaid.pdf")



