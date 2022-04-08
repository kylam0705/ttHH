import pandas
import awkward
import numpy
import matplotlib.pyplot as plt
from yahist import Hist1D
import argparse
import json
import math
import scipy.integrate
from numpy import nan

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

#New Columns in Awkward Array
events_awkward["Photon_mvaID"] = awkward.concatenate(
	[
		awkward.unflatten(events_awkward.LeadPhoton_mvaID, 1), 
		awkward.unflatten(events_awkward.SubleadPhoton_mvaID, 1)
	],
	axis = 1
)
events_awkward["MaxPhoton_mvaID"] = awkward.max(events_awkward.Photon_mvaID, axis = 1)
events_awkward["MinPhoton_mvaID"] = awkward.min(events_awkward.Photon_mvaID, axis = 1)

#Data
data = events_json["sample_id_map"]["Data"]
events_data = events[events["process_id"] == data]
events_data["MinPhoton_mvaID"] = events_data[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].min(axis=1)
events_data["MaxPhoton_mvaID"] = events_data[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].max(axis=1)
data_in_sideband_cut = events_data[events_data["MinPhoton_mvaID"] < args.sideband_cut]
#print(data_in_sideband_cut.columns)

#Data in Awkward Array
events_data_ak = events_awkward[events_awkward["process_id"] == data]
#events_data_ak["Photon_mvaID"] = awkward.concatenate(
#	[
#		awkward.unflatten(events_data_ak.LeadPhoton_mvaID, 1),
#		awkward.unflatten(events_data_ak.SubleadPhoton_mvaID, 1)
#	],
#	axis = 1
#)
#events_data_ak["MaxPhoton_mvaID"] = awkward.max(events_data_ak.Photon_mvaID, axis = 1)
#events_data_ak["MinPhoton_mvaID"] = awkward.min(events_data_ak.Photon_mvaID, axis = 1)
data_in_sideband_ak = events_data_ak[events_data_ak["MinPhoton_mvaID"] < args.sideband_cut]
#print("data columns", events_data_ak.fields)
#print("Max data", events_data_ak.MaxPhoton_mvaID)

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
#events_GJets["Lead_Fake_Photon_mvaID"] = fake_lead["LeadPhoton_mvaID"]
#fake_lead_id = events_GJets["Lead_Fake_Photon_mvaID"]
fake_sublead = events_GJets[events_GJets["SubleadPhoton_genPartFlav"] == 0]
fake_sublead_id = fake_sublead["SubleadPhoton_mvaID"]
#events_GJets["Sublead_Fake_Photon_mvaID"] = fake_sublead["SubleadPhoton_mvaID"]
#fake_sublead_id = events_GJets["Sublead_Fake_Photon_mvaID"]
#Fake ID
fake_id = pandas.concat([fake_lead_id, fake_sublead_id]) #Creates an array of fake photons to be inserted into the histogram h_fake

#Min & Max:
#Should I first make these fake_lead_id/fake_sublead_id into columns like events_GJets['LeadFakePhoton_mvaID'] = fake_lead["LeadPhoton_mvaID"]  and then replace fake_lead_id in the next line with "Lead_Fake_Photon_mvaID" etc 
#events_GJets["MinFakePhoton_mvaID"] = events_GJets[["Lead_Fake_Photon_mvaID", "Sublead_Fake_Photon_mvaID"]].min(axis=1) 
#events_GJets["MaxFakePhoton_mvaID"] = events_GJets[["Lead_Fake_Photon_mvaID", "Sublead_Fake_Photon_mvaID"]].max(axis=1) 

#print("min_fake_id", min(fake_id))
#print("max_fake_id", max(fake_id))

#Making the Fake PDF Histogram
f = plt.figure()

def round_down(n, decimals):
	multiplier = 10 ** decimals 
	rounded_number = math.floor(n * multiplier) / multiplier
	return rounded_number

lower_range = float(round_down(args.sideband_cut, 1))
n_bins = int((1-lower_range)/0.05)

h_fake = Hist1D(fake_id, bins = "%d, %.1f, 1" %(n_bins, lower_range), overflow=False) #Histogram of fake photons from fake_id array. 40 bins are set so that each bin covers a 0.5 id score range
h_fake = h_fake.normalize()

h_fake_all = Hist1D(fake_id, bins = "100,-1,1") #This one is used later for taking the integrals
h_fake_all = h_fake_all.normalize()

h_weight = Hist1D(fake_id, bins = "40,-1,1")
h_weight = h_weight.normalize()

#Random.Choice Inputs:
Last_Bin = n_bins
First_Bin = 0

p_bins = h_fake.counts[First_Bin:Last_Bin]
p = p_bins/numpy.sum(p_bins) #p-value in the random.choice function

#Making the PDF Histogram
fake_photons_pdf = numpy.random.choice(a=n_bins, size = data_in_sideband_cut.size, p=p) #fake_photons is an array of integers that identifies the bin. I need to convert the identified bins to idmva scores in the [sideband_cut,1] range ie new_pdf
#print("min fake_photons_pdf", min(fake_photons_pdf))
print("fake_photons_pdf", fake_photons_pdf)

hist_idmva_low = {}
for i in range(h_weight.nbins): 
	hist_idmva_low[i] = round(h_weight.edges[i],2) #The keys in this dictionary are the bin numbers, the values are the lower bin edge score

new_pdf = []
new_pdf = [hist_idmva_low[key] for key in fake_photons_pdf] #This array is the lower bin edge scores of the fake_photon_pdf array of bin numbers
new_pdf_array = numpy.array(new_pdf)
print("new_pdf_array", new_pdf_array)
print("Min pdf array", min(new_pdf_array))

low = new_pdf_array
high = new_pdf_array + round(h_fake.bin_widths[1],2)
size = new_pdf_array.size
plotted_pdf = numpy.random.uniform(low = low, high= high, size = new_pdf_array.size) #This is the new array that needs to be plotted 
print("min plotted_pdf: ", min(plotted_pdf))

h_attempt = Hist1D(plotted_pdf, bins = "%d,%.1f,1" %(n_bins, lower_range), overflow=False)
h_attempt = h_attempt.normalize()

#Plotted
h_attempt.plot(label = "Random Function", color = 'blue')
h_fake.plot(label = "Fake Photons from GJets", color = 'orange')

#Labels/Aesthetics
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8, 0.2, 0.2))
plt.yscale("log")
plt.xlabel("IDMVA Score")
plt.ylabel("Normalized Events")

plt.title("Fake Photon IDMVA in GJets")
#plt.text(0,1, "CMS Preliminary", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
#plt.text(1,1, "137 fb$^{-1}$ (13TeV)", horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes)

plt.show()
f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_mvaid.pdf")

#Reweighing Events
##Random Choice
random_choice_function = numpy.random.choice(a=40, size = data_in_sideband_cut.size, p = h_weight.counts)
rescaled_events = [hist_idmva_low[key] for key in random_choice_function]
f_rescaled_events_array = numpy.array(rescaled_events)
size_new = f_rescaled_events_array.size
s_rescaled_events_array = f_rescaled_events_array + numpy.random.uniform(low = 0, high = round(h_weight.bin_widths[1],3), size = size_new) #This array allocates a score to an event based on the bin value
print("first rescaled_events_array", f_rescaled_events_array)
print("second rescaled_events_array", s_rescaled_events_array)

#Histograms
fig = plt.figure()
h_first = Hist1D(f_rescaled_events_array, bins = "100,-1,1")
h_first = h_first.normalize()
h_second = Hist1D(s_rescaled_events_array, bins = "100,-1,1")
h_second = h_second.normalize()

h_first.plot(label = "Lower Bin Score", color = 'blue', histtype = 'stepfilled', alpha = 0.8)
h_second.plot(label = "Bin Score + random.uniform", color = 'orange', histtype = 'stepfilled', alpha = 0.8)

plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8, 0.2, 0.2))
plt.yscale("log")
plt.xlabel("IDMVA Score")
plt.ylabel("Normalized Events")

plt.show()
fig.savefig("/home/users/kmartine/public_html/plots/Fall_2021/rescaled_events.pdf")

hist_idmva_low_scores = {}
for i in range(h_fake_all.nbins): 
	hist_idmva_low_scores[i] = round(h_second.edges[i],2) #The keys in this dictionary are the bin numbers, the values are the lower bin edge score

#Bounds of Integral
def find_nearest(array, value):
	val = numpy.ones_like(array)*value
	idx = (numpy.abs(array-val)).argmin()
	return array[idx], idx

#print("length of data_in_sideband_cut", len(data_in_sideband_cut))
#print("length of ak data in sideband_cut", len(data_in_sideband_ak))
#print("data_in_sideband_ak.MaxPhoton_mvaID", data_in_sideband_ak.MaxPhoton_mvaID)

unneeded, sideband_cut_bound = find_nearest(h_second.bin_centers, args.sideband_cut)
omega = numpy.ones(len(data_in_sideband_ak))
for i in range(len(data_in_sideband_ak)):
	val, num_max_bound = find_nearest(h_second.bin_centers, data_in_sideband_ak.MaxPhoton_mvaID[i])
	numerator = sum(h_fake_all.counts[sideband_cut_bound : num_max_bound])
	denominator = sum(h_fake_all.counts[0 : sideband_cut_bound])
	omega[i] = numerator / denominator
#print(omega,"omega")
#print(num_max_bound, "num_max_bound")
#print(numerator, "numerator")

#New Weights
original_weight = data_in_sideband_cut["weight_central"]
new_weight = original_weight * omega
#print("original_weight", original_weight)
#print("new_weight", new_weight)

print(events_awkward.fields, "events_awkward.fields")

#Total Normational
n_total_bkg = 0
other_bkgs = ["Diphoton", "HH_ggbb", "HHggTauTau", "TTGG", "TTGamma", "TTJets", "VBFH_M125", "VH_M125", "WGamma", "ZGamma", "ggH_M125", "ttH_M125"]
for bkg in other_bkgs: 
	events_bkg =  events_awkward[events_awkward["process_id"] == events_json["sample_id_map"][bkg]]
	n_bkg = awkward.sum(events_bkg.weight_central)
	n_total_bkg += n_bkg

n_data = sum(events_data_ak.weight_central)

n_GJets = n_data - n_total_bkg
scale_factor = sum(events_awkward.weight_central) / n_GJets
print(scale_factor)

total_normal_weight = new_weight * scale_factor
print(total_normal_weight, "total_normal_weight")

print(len(plotted_pdf), "plotted_pdf")
print(len(data_in_sideband_ak), "length of events reinserted into preselection")

#Concat to new parquet file
#events_dd should have all the same fields as events_awkward with the fields "MinPhoton_mvaID", "process_id", and "weight_central" (per events and overall normalization factor) updated
#events_dd = awkward.Array([
#	{"MinPhoton_mvaID":plotted_pdf,
#	"process_id" : '21',
#	"weight_central":new_weight}
#])

#print(events_dd, "events_dd")

#events_all = awkward.concatenate([events_awkward, events_dd])
#print(events_all, "events_all")
#awkward.to_parquet(events_all, "merged_nominal_with_data_drives_qcdgjets.parquet")








