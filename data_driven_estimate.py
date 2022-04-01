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

args = parser.parse_args()

# 1. Load events and process id map
events = awkward.from_parquet("/home/users/smay/public_html/forKyla/merged_nominal.parquet")

with open("/home/users/smay/public_html/forKyla/summary.json", "r") as f_in:
    process_id_map = json.load(f_in)["sample_id_map"]


# 2. Create min/max photon ID MVA fields
# First create a new field with both the lead and sublead mva ids
events["Photon_mvaID"] = awkward.concatenate(
        [awkward.unflatten(events.LeadPhoton_mvaID, 1), awkward.unflatten(events.SubleadPhoton_mvaID, 1)],
        axis = 1
)
# Then we take the min/max of these in each event
events["MaxPhoton_mvaID"] = awkward.max(events.Photon_mvaID, axis = 1)
events["MinPhoton_mvaID"] = awkward.min(events.Photon_mvaID, axis = 1)

# 2.1 Split events into preselection events and sideband events
presel_events = events[(events.MinPhoton_mvaID > args.sideband_cut) & (events.MaxPhoton_mvaID > args.sideband_cut)]
sideband_events = events[(events.MinPhoton_mvaID < args.sideband_cut) & (events.MaxPhoton_mvaID > args.sideband_cut)]

# 2.2 Split events into preselection events and sideband events
sideband_events_data = sideband_events[sideband_events.process_id == process_id_map["Data"]]

# check that this did what we expected by printing out the first few events
for i in range(3):
    print("Lead mva ID", events.LeadPhoton_mvaID[i])
    print("Sublead mva ID", events.SubleadPhoton_mvaID[i])
    print("Max mva ID", events.MaxPhoton_mvaID[i])
    print("Min mva ID", events.MinPhoton_mvaID[i])

# 3. Derive pdf for fake photons from GJets MC
# 3.1 Select GJets events
gjets_cut = (events.process_id >= process_id_map["GJets_HT-40To100"]) & (events.process_id <= process_id_map["GJets_HT-600ToInf"])
events_gjets = events[gjets_cut]

# now get prompt and fake photons separately
prompt_id = awkward.concatenate([
            events_gjets[events_gjets.LeadPhoton_genPartFlav == 1].LeadPhoton_mvaID,
            events_gjets[events_gjets.SubleadPhoton_genPartFlav == 1].SubleadPhoton_mvaID
        ]
)

fake_id = awkward.concatenate([
            events_gjets[events_gjets.LeadPhoton_genPartFlav == 0].LeadPhoton_mvaID,
            events_gjets[events_gjets.SubleadPhoton_genPartFlav == 0].SubleadPhoton_mvaID
        ]
)

# check that this did we what expected
# prompt photons should have higher mva ID than fake photons on average
print("Mean photon ID MVA value for prompt photons: ", awkward.mean(prompt_id))
print("Mean photon ID MVA value for fake photons: ", awkward.mean(fake_id))

# 3.2 Derive the fake pdf histogram

# 3.3 Make plots
f = plt.figure()

# Make a function to determine the number of bins and the lower edge of the histogram
def round_down(n, decimals):
	multiplier = 10**decimals 
	rounded_number = math.floor(n * multiplier) / multiplier
	return rounded_number 

lower_range = float(round_down(args.sideband_cut, 1))
n_bins = int((1-lower_range)/0.05) 

#  Plots
h_fake_sideband_cut_to_one = Hist1D(fake_id, bins = "%d, %.1f, 1" %(n_bins, lower_range), overflow=False)
h_fake_sideband_cut_to_one = h_fake.normalize #This is a plot of the fake photons from GJets MC

h_fake_minus_one_to_one = Hist1D(fake_id, bins = "100,-1,1", ovrflow = False)
h_fake_minus_one_to_one = h_fake_minus_one_to_one.normalize()

# 3.4 Define a function for generating an arbitrary number of events from the fake pdf
def generate_from_fake_pdf(fake_pdf, n):
    """
    Returns a 1d array of length n of values generated from a binned probability distribution `fake_pdf`
    """
    # your implementation from old code here

# 4. Add data-driven events into preselection array
# 4.1 Set their min photon ID MVA score equal to the values randomly generated according to the fake PDF
generated_photon_id_scores = awkward.ones_like(sideband_events_data.LeadPhoton_mvaID) # dummy array of all 1's, you should update with your function for generating the scores
generated_photon_id_scores = #Insert function that generates the scores
sideband_events_data["MinPhoton_mvaID"] = generated_photon_id_scores

# 4.2 Apply the per-event and overall normalization factors to the central weight of data sideband events
# 4.2.A Find the per-event scale factor
def find_nearest(array,value): #This array will give you an array of and the bins associated with the integral bounds
	val = numpy.ones_like(array)*value
	idx = (numpy.abs(array-val)).argmin()
	return array[idx], idx

unneeded, sideband_cut_bound = find_nearest("histogram of the events with random scores from (-1,1)", args.sideband_cut)
omega = numpy.ones(len(sideband_events_data)) #Omega is the per-event scale factor
for i in range(len(sideband_events_data): 
	val, num_max_bound = find_nearest("histogram mentioned above", sideband_events_data.MaxPhoton_mvaID[i])
	numerator = sum(h_fake_minus_one_to_one.counts[sideband_cut_bound:num_max_bound])
	denominator = sum(h_fake_minus_one_to_one.counts[0:sideband_cut_bound]
	omega[i] = numerator / denominator

# check that omega looks correct for the first few events
for i in range(5):
	print("Per-event scale factor:", omega[i])

#4.2.B Apply the per-event normalization 
sideband_events_data["weight_central"] = sideband_events_data.weight_central * omega

# check that these events are scaled correctly
for i in range(5):
	print("Events weight scaled by omega:", sideband_events_data.weight_central[i])

# 4.2.C Find the overall normalization factor
n_total_bkg = 0 
other_bkgs = ["Diphoton", "HH_ggbb", "HHggTauTau", "TTGG", "TTGamma", "TTJets", "VBFH_M125", "VH_M125", "WGamma", "ZGamma", "ggH_M125", "ttH_M125"]
for bkg in other_bkgs:
	events_bkg = events[events["process_id"] == process_id_map[bkg]]
	n_bkg = awkward.sum(events_bkg.weight_central)
	n_total_bkg += n_bkg

n_data 
n_GJets 
total_norm_factor = sum(events.weight_central) / n_GJets

# 4.3.D Apply the overall normalization factor
sideband_events_data["weight_central"] = sideband_events_data.weight_central * total_norm_factor



# 5. Append the data sideband events (which now have MinPhoton_mvaID values in the preselection range) to the preselection events
# 5.1 Give them a new process_id value
sideband_events_data["process_id"] = awkward.ones_like(sideband_events_data.process_id) * 21 # this is an array of all 21's

presel_and_data_sideband_events = awkward.concatenate(
        [presel_events, sideband_events_data]
)

# 6. Save these to a new parquet file for further analysis
awkward.to_parquet(presel_and_data_sideband_events, "presel_with_dd_estimate.parquet")





