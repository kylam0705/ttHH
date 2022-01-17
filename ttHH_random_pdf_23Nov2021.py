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

#Data
data = events_json["sample_id_map"]["Data"]
events_data = events[events["process_id"] == data]
events_data["MinPhoton_mvaID"] = events_data[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].min(axis=1)
data_in_sideband_cut = events_data[events_data["MinPhoton_mvaID"] < args.sideband_cut]
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
#print("min_fake_id", min(fake_id))
fake_id_events = pandas.concat([fake_lead, fake_sublead])

#More Columns in the Dataframe
#min_value_series = events['LeadPhoton_mvaID','SubleadPhoton_mvaID'].min(axis=1)
#events["MinPhoton_mvaID"] = min_value_series
#events_photon_sideband_cut = events[events["MinPhoton_mvaID"] > -0.67]
#events_photon_preselection_cut = events[events["max_pho_idmva"] > -0.67]

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

hist_idmva_low = {}
for i in range(h_weight.nbins): 
	hist_idmva_low[i] = round(h_weight.edges[i],2) #The keys in this dictionary are the bin numbers, the values are the lower bin edge score

new_pdf = []
new_pdf = [hist_idmva_low[key] for key in fake_photons_pdf] #This array is the lower bin edge scores of the fake_photon_pdf array of bin numbers
new_pdf_array = numpy.array(new_pdf)
#print("Min pdf array", min(new_pdf_array))

low = new_pdf_array
high = new_pdf_array + round(h_fake.bin_widths[1],2)
size = new_pdf_array.size
plotted_pdf = numpy.random.uniform(low = low, high= high, size = new_pdf_array.size) #This is the new array that needs to be plotted 
#print("min plotted_pdf: ", min(plotted_pdf))

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
rescaled_events_array = numpy.array(rescaled_events)
low_new = rescaled_events_array
high_new = rescaled_events_array + 0.05
size_new = rescaled_events_array.size
random_pdf = numpy.random.uniform(low = low_new, high = high_new, size = size_new)
random_pdf = random_pdf[numpy.logical_not(numpy.isnan(random_pdf))]

sideband_cut_bound = round(args.sideband_cut, 2)

upper_limit_num = max(random_pdf)
lower_limit_denom = min(random_pdf)

num_array = [event for event in random_pdf if event >= sideband_cut_bound]
denom_array = [event for event in random_pdf if event <= sideband_cut_bound]
numerator = round(numpy.sum(num_array), 4)
denominator = abs(round(numpy.sum(denom_array), 4))
omega = numerator/denominator

#print("num_array", num_array)
#print("denom_array", denom_array)
print("numerator", numerator)
print("denominator", denominator)
print("omega", omega)

#original_weight = events[events["weight_central_initial"]]
original_weight = fake_id_events["weight_central"]
print("original_weight", original_weight)
new_weight = original_weight * omega
print("new_weight", new_weight)



