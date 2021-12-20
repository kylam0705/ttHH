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
h_fake = Hist1D(fake_id, bins = "40,-1,1") #Histogram of fake photons from fake_id array
h_fake = h_fake.normalize()
#P-values:
Beginning = (1.0-abs(min(fake_id)))*20
Beginning = int(Beginning)
print(Beginning)
rounded_number = 0.05 * round(abs(args.sideband_cut) / 0.05)
if args.sideband_cut <= 0.0: 
	Ending = (1- rounded_number) * 20
	Ending = int(Ending)
if args.sideband_cut > 0.0: 
	Ending = 20 + ((1- rounded_number)*20)
	Ending = int(Ending)
print(Ending)
p_bins = h_fake.counts[Beginning:Ending]
#p_bins = h_fake.counts[6:20] #Is there a way to change the code so that these ranges get updates with the sideband_cut? 
p = p_bins/numpy.sum(p_bins) #p-value in the random.choice function

#PDF Function
fake_photons_pdf = numpy.random.choice(a=14, size = sideband_cut.size, p=p) #fake_photons is an array of integers in the wrong range. I need to convert it to floats in the [sideband_cut,1] rangei

#Rescaling Function
oMin = min(fake_photons_pdf)
oMax = max(fake_photons_pdf)
nMin = args.sideband_cut
nMax = 1.0

def remap(x, oMin, oMax, nMin, nMax):
	#range check
	if oMin == oMax:
		print("Warning: Zero input range")
		return None
	if nMin == nMax:
		print("Warning: Zer output range")
		return None

	#Check reversed input range
	reverse_Input = False
	old_min = min(oMin, oMax)
	old_max = max(oMin, oMax)
	if not old_min == old_max: 
		reverse_Input = True

	#Check reversed output range
	reverse_output = False
	new_min = min(nMin, nMax)
	new_max = max(nMin,nMax)
	if not new_min == new_max:
		reverse_output = True

	portion = ((x-old_min)*(new_max - new_min))/(old_max-old_min)
	if reverse_Input: 
		portion = ((old_max - x)*(new_max-new_min))/(old_max-old_min)
	result = portion + new_min
	if reverse_output: 
		result = new_max - portion

	return result
	#print(result)
attempt_array = remap(fake_photons_pdf, oMin, oMax, nMin, nMax)
#print('...')
#print(remap(fake_photons_pdf, oMin, oMax, nMin, nMax))

#Plotting
h_pdf = Hist1D(fake_photons_pdf, bins = "40,-1,1") #This is the pdf that needs to be plotted in a histogram
h_attempt = Hist1D(attempt_array, bins = "40, -1,1")

#h_attempt.plot(hisstype="stepfillled", alpha = 0.8, label = "Attempt of Random Function Fixed")
h_attempt.plot(alpha = 0.8, label = "Attempt of Random Function Fixed")
h_pdf.plot(histtype="stepfilled", alpha = 0.8, label = "Random Function")
h_fake.plot(histtype="stepfilled", alpha = 0.8, label = "Fake Photons from GJets")

plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8, 0.2, 0.2))
plt.yscale("log")
plt.title("Fake Photon IDMVA in GJets")
plt.xlabel("IDMVA Score")
plt.ylabel("Events")
plt.show()
#f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_GJets.pdf")
f.savefig("/home/users/kmartine/public_html/plots/Fall_2021/fake_photons_mvaid.pdf")



