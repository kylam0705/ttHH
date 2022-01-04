import pandas
import awkward
import numpy
import matplotlib.pyplot as plt
from yahist import Hist1D
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
	"--sideband_cut",
	required = True,
	default = None,
	type = float,
	help = "Places a cut to establish the sideband definition")
#parser.add_argument(
#	"--input_parquet",
#	required = False,
#	default = None, 
#	help = "Path to parquet file")
args = parser.parse_args()

#Reading Files
#Parquet
#events_awkward = awkward.from_parquet(args.input_parquet)
events_awkward = awkward.from_parquet("/home/users/smay/public_html/forKyla/merged_nominal.parquet")
events = awkward.to_pandas(events_awkward)
#JSON
json_file = open("/home/users/smay/public_html/forKyla/summary.json")
events_json = json.load(json_file)
json_file.close()

#Data
data = events_json["sample_id_map"]["Data"]
events_data = events[events["process_id"] == data]
events_data["MinPhoton_mvaID"] = events_data[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].min(axis=1)
sideband_cut = events_data[events_data["MinPhoton_mvaID"] < args.sideband_cut]
#print(sideband_cut.size)

#Gamma + Jets Process:
GJets_min = events_json["sample_id_map"]["GJets_HT-40To100"]
GJets_max = events_json["sample_id_map"]["GJets_HT-600ToInf"]
events_GJets = events[(events["process_id"] >= GJets_min) & (events["process_id"] <= GJets_max)]
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

#Making the Fake PDF Histogram
f = plt.figure()
h_fake = Hist1D(fake_id, bins = "40,-1,1") #Histogram of fake photons from fake_id array. 40 bins are set so that each bin covers a 0.5 id score range
h_fake = h_fake.normalize()

#Random.Choice Inputs:
Beginning = (1.0-abs(min(fake_id)))*20 #If the number of bins change then the 20's and 0.05's will also have to change
Beginning = int(Beginning)
rounded_number = 0.05 * round(abs(args.sideband_cut) / 0.05)
if args.sideband_cut <= 0.0: 
	Ending = (1- rounded_number) * 20
	Ending = int(Ending)
if args.sideband_cut > 0.0: 
	Ending = 20 + ((1- rounded_number)*20)
	Ending = int(Ending)
p_bins = h_fake.counts[0:Ending]
#p_bins = h_fake.counts[Beginning:Ending] 
p = p_bins/numpy.sum(p_bins) #p-value in the random.choice function
#a = Ending - Beginning

#Making the PDF Histogram
#print("Beginning", Beginning)
#print("Ending", Ending)

fake_photons_pdf = numpy.random.choice(a=20, size = sideband_cut.size, p=p) #fake_photons is an array of integers that identifies the bin. I need to convert the identified bins to idmva scores in the [sideband_cut,1] range ie new_pdf
fake_photons_pdf = fake_photons_pdf + Ending #Because the array begins at 0, they might get updated (ie 0 in the array could be the 5th bin assuming the bins start at a nonzero number) 
#print(fake_photons_pdf)
#print("max Fake photons", max(fake_photons_pdf))
hist_idmva_low = {}
for i in range(h_fake.nbins): 
	hist_idmva_low[i] = round(h_fake.edges[i],2) #The keys in this dictionary are the bin numbers, the values are the lower bin edge score

new_pdf = []
new_pdf = [hist_idmva_low[key] for key in fake_photons_pdf] #This array is the lower bin edge scores of the fake_photon_pdf array of bin numbers
new_pdf_array = numpy.array(new_pdf)

low = new_pdf_array
high = new_pdf_array + round(h_fake.bin_widths[1],2)
size = new_pdf_array.size
plotted_pdf = numpy.random.uniform(low = low, high= high, size = size) #This is the new array that needs to be plotted

h_attempt = Hist1D(plotted_pdf, bins = "40, -1,1")
h_attempt = h_attempt.normalize()

#Making the Fake PDF Subplot Histogram 
h_subplot = Hist1D(fake_id, bins = "20,0,1", overflow=False)
h_subplot = h_subplot.normalize()

#Subplots: 
fig, (ax1,ax2) = plt.subplots(2, figsize=(8,6), gridspec_kw=dict(height_ratios=[3,1]))

#Plotted
h_attempt.plot(ax=ax1,histtype = "stepfilled", alpha = 0.8, label = "Random Function", color = 'blue')
h_fake.plot(ax=ax1,histtype="stepfilled", alpha = 0.8, label = "Fake Photons from GJets", color = 'orange')
h_subplot.plot(ax=ax2, histtype="stepfilled", alpha = 0.8, color = 'orange')

#Labels/Aesthetics
ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8, 0.2, 0.2))
ax1.set_yscale("log")
ax1.set_xlabel("IDMVA Score")
ax1.set_ylabel("Normalized Events")

ax2.set_yscale("log")
ax2.set_xlabel("IDMVA Score")
ax2.set_ylabel("Normalized Events")

fig.suptitle("Fake Photon IDMVA in GJets")
plt.text(0,1, "CMS Preliminary", horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes)
plt.text(1,1, "137 fb$^{-1}$ (13TeV)", horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes)

plt.show()
#f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_GJets.pdf")
fig.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_mvaid.pdf")



