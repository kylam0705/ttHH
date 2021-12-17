import numpy
import pandas
import awkward
import argparse

events_awkward = awkward.from_parquet("/home/users/smay/public_html/forKyla/merged_nominal.parquet")
events = awkward.to_pandas(events_awkward)

events["MinPhoton_mvaID"] = events[['LeadPhoton_mvaID','SubleadPhoton_mvaID']].min(axis = 1)

parser = argparse.ArgumentParser()
parser.add_argument(
	"--target_signal_eff",
	required = True,
	default = None,
	type = float,
	help = "Determines the percentage of efficiency allowed to pass the cut")
args = parser.parse_args()

def quantiles_to_idmva_score(n_quantiles, MinPhoton_mvaID):
	sorted_mva = numpy.sort(MinPhoton_mvaID)
	map = {}

	for i in range(n_quantiles):
		idx = int((float(i+1) / float(n_quantiles)) * len(sorted_mva)) - 1
		quantile = float(i) / float(n_quantiles)
		mva_score = sorted_mva[idx]
		map[quantile] = mva_score
	#print(map)
	sideband_cut = 1 - args.target_signal_eff
	sideband_cut = float(round(sideband_cut, 2))
	print(map[sideband_cut]) 

quantiles_to_idmva_score(100, events["MinPhoton_mvaID"])


