import pandas
import awkward
import numpy
import matplotlib.pyplot as plt
from yahist import Hist1D
from yahist.utils import plot_stack
import argparse
import json
import math
import scipy.integrate
from numpy import nan

#Parser Arguments
#parser = argparse.ArgumentParser()
#parser.add_argument(
#       "--input_parquet",
#       required = False,
#       default = None, 
#       help = "Path to parquet file")
#args = parser.parse_args()

# 1. Load events and process id map
#Parquet File
#events_awkward = awkward.from_parquet(args.input_parquet)
events = awkward.from_parquet("/home/users/kmartine/public_html/ttHH_python_codes/presel_with_dd_estimate.parquet")

#JSON File
json_file = open("/home/users/kmartine/public_html/ttHH_python_codes/summary.json")
events_json = json.load(json_file)
json_file.close()

# 2. Make a function to make a log plot of signal, MC, and data with ratio panel below 
def make_log_ratio_plot(data, mc, mc_weight, gg_mc, gg_mc_weight, gjets_mc, gjets_mc_weight, hh_ggbb_mc, hh_ggbb_mc_weight, ttgg_mc, ttgg_mc_weight, ttg_mc, ttg_mc_weight, vh_mc, vh_mc_weight, wgamma_mc, wgamma_mc_weight, zgamma_mc, zgamma_mc_weight, dd_gjets_qcd_mc, dd_gjets_qcd_mc_weight, tthh_ggbb, tthh_ggbb_weight, tthh_ggWW, tthh_ggWW_weight, tthh_ggTauTau, tthh_ggTauTau_weight, log_save_name, **kwargs): 

	normalize = kwargs.get("normalize", False)
	x_label = kwargs.get("x_label", None)
	y_label = kwargs.get("y_label", "Events" if not normalize else "Fraction of Events")
	rat_label = kwargs.get("rat_label", "Data/MC")
	title = kwargs.get("title", None)
	y_lim = kwargs.get("y_lim", None)
	x_lim = kwargs.get("x_lim", None)
	rat_lim = kwargs.get("rat_lim", None)
	log_y = kwargs.get("log_y", True)
	bins = kwargs.get("bins", None)
	x_label = kwargs.get("x_label", None)

	#Data
	h_data = Hist1D(data, bins = bins)

	#MC BKG
	h_mc = Hist1D(mc, bins = bins, color = "C3", weights = mc_weight, label = "All Background (MC)")
	h_gg_mc = Hist1D(gg_mc, bins = bins, color = "lightblue", weights = gg_mc_weight, label = "\u03B3\u03B3")
	h_gjets_mc = Hist1D(gjets_mc, bins = bins, color = "green", weights = gjets_mc_weight, label = "GJets")
	h_hh_ggbb_mc = Hist1D(hh_ggbb_mc, bins = bins, color = "orange", weights = hh_ggbb_mc_weight, label = "HH->ggbb")
	h_ttgg_mc = Hist1D(ttgg_mc, bins = bins, color = "red", weights = ttgg_mc_weight, label = "TTGG")
	h_ttg_mc = Hist1D(ttg_mc, bins = bins, color = "purple", weights = ttg_mc_weight, label = "TTGamma")
	h_vh_mc = Hist1D(vh_mc, bins = bins, color = "pink", weights = vh_mc_weight, label = "VH")
	h_wgamma_mc = Hist1D(wgamma_mc, bins = bins, color = "cyan", weights = wgamma_mc_weight, label = "W\u03B3")
	h_zgamma_mc = Hist1D(zgamma_mc, bins = bins, color = "olive", weights = zgamma_mc_weight, label = "Z\u03B3")
	h_dd_gjets_qcd_mc = Hist1D(dd_gjets_qcd_mc, bins = bins, color = "brown", weights = dd_gjets_qcd_mc_weight, label = "DD Description GJets and QCD")
	
	#Signal 
	h_tthh_ggbb = Hist1D(tthh_ggbb, bins = bins, color = "lawngreen", weights = tthh_ggbb_weight, label = "ttHH ->\u03B3 \u03B3bb (x$10^3$)")
	h_tthh_ggWW = Hist1D(tthh_ggWW, bins = bins, color = "deepskyblue", weights = tthh_ggWW_weight, label = "ttHH ->\u03B3 \u03B3WW (x$10^3$)")
	h_tthh_ggTauTau = Hist1D(tthh_ggTauTau, bins = bins, color = "indigo", weights = tthh_ggTauTau_weight, label = "ttHH ->\u03B3 \u03B3 \u03C4 \u03C4 (x$10^3$)")

	fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize = (8,6), gridspec_kw = dict(height_ratios=[3,1]))
	plt.grid()

	hist_stack = [h_hh_ggbb_mc, h_vh_mc, h_ttgg_mc, h_zgamma_mc, h_wgamma_mc, h_ttg_mc, h_gg_mc, h_gjets_mc, h_dd_gjets_qcd_mc]
	hist_stack = sorted(hist_stack, key = lambda x:x.integral)

	h_data.plot(ax = ax1, alpha = 0.8, color = "black", errors=True, label = "Data")
	plot_stack(hist_stack, histtype = "stepfilled", ax = ax1, alpha = 0.8)
	h_tthh_ggbb.plot(ax=ax1, alpha = 0.8)
	h_tthh_ggWW.plot(ax=ax1, alpha = 0.8)
	h_tthh_ggTauTau.plot(ax=ax1, alpha = 0.8)

	ratio = h_data / h_mc
	ratio.plot(ax=ax2, errors=True, color="black")

	if log_y: 
		ax1.set_yscale("log")

	if x_label is not None:
		ax2.set_xlabel(x_label)

	if y_label is not None: 
		ax1.set_ylabel(y_label)

	if title is not None: 
		ax1.set_title(title)

	if y_lim is not None: 
		ax1.set_ylim(y_lim)

	if rat_lim is not None: 
		ax2.set_ylim(rat_lim)

	if x_lim is not None: 
		ax1.set_xlim(x_lim) 

	if rat_label is not None: 
		ax2.set_ylabel("Data/MC")

	ax2.set_ylim([0.0,2.0])
	plt.text(0,1, "CMS Preliminary", horizontalalignment = 'left', verticalalignment = 'bottom', transform = ax1.transAxes)
	plt.text(1,1, "137 fb$^{-1}$ (13 TeV)", horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax1.transAxes)

	plt.savefig(log_save_name)

# 3. Make a linear plot same as above
def make_linear_ratio_plot(data, mc, mc_weight, gg_mc, gg_mc_weight, gjets_mc, gjets_mc_weight, hh_ggbb_mc, hh_ggbb_mc_weight, ttgg_mc, ttgg_mc_weight, ttg_mc, ttg_mc_weight, vh_mc, vh_mc_weight, wgamma_mc, wgamma_mc_weight, zgamma_mc, zgamma_mc_weight, dd_gjets_qcd_mc, dd_gjets_qcd_mc_weight, tthh_ggbb, tthh_ggbb_weight, tthh_ggWW, tthh_ggWW_weight, tthh_ggTauTau, tthh_ggTauTau_weight, linear_save_name, **kwargs):

	normalize = kwargs.get("normalize", False)
	x_label = kwargs.get("x_label", None)
	y_label = kwargs.get("y_label", "Events" if not normalize else "Fraction of Events")
	rat_label = kwargs.get("rat_label", "Data/MC")
	title = kwargs.get("title", None)
	y_lim = kwargs.get("y_lim", None)
	x_lim = kwargs.get("x_lim", None)
	rat_lim = kwargs.get("rat_lim", None)
	log_y = kwargs.get("log_y", False)
	bins = kwargs.get("bins", None)
	x_label = kwargs.get("x_label", None)

	#Data
	h_data = Hist1D(data, bins = bins)

	#MC BKG
	h_mc = Hist1D(mc, bins = bins, color = "C3", weights = mc_weight, label = "All Background (MC)")
	h_gg_mc = Hist1D(gg_mc, bins = bins, color = "lightblue", weights = gg_mc_weight, label = "\u03B3\u03B3")
	h_gjets_mc = Hist1D(gjets_mc, bins = bins, color = "green", weights = gjets_mc_weight, label = "GJets")
	h_hh_ggbb_mc = Hist1D(hh_ggbb_mc, bins = bins, color = "orange", weights = hh_ggbb_mc_weight, label = "HH->ggbb")
	h_ttgg_mc = Hist1D(ttgg_mc, bins = bins, color = "red", weights = ttgg_mc_weight, label = "TTGG")
	h_ttg_mc = Hist1D(ttg_mc, bins = bins, color = "purple", weights = ttg_mc_weight, label = "TTGamma")
	h_vh_mc = Hist1D(vh_mc, bins = bins, color = "pink", weights = vh_mc_weight, label = "VH")
	h_wgamma_mc = Hist1D(wgamma_mc, bins = bins, color = "cyan", weights = wgamma_mc_weight, label = "W\u03B3")
	h_zgamma_mc = Hist1D(zgamma_mc, bins = bins, color = "olive", weights = zgamma_mc_weight, label = "Z\u03B3")
	h_dd_gjets_qcd_mc = Hist1D(dd_gjets_qcd_mc, bins = bins, color = "brown", weights = dd_gjets_qcd_mc_weight, label = "DD Description GJets and QCD")

	#Signal 
	h_tthh_ggbb = Hist1D(tthh_ggbb, bins = bins, color = "lawngreen", weights = tthh_ggbb_weight, label = "ttHH ->\u03B3 \u03B3bb (x$10^5$)")
	h_tthh_ggWW = Hist1D(tthh_ggWW, bins = bins, color = "deepskyblue", weights = tthh_ggWW_weight, label = "ttHH ->\u03B3 \u03B3WW (x$10^5$)")
	h_tthh_ggTauTau = Hist1D(tthh_ggTauTau, bins = bins, color = "indigo", weights = tthh_ggTauTau_weight, label = "ttHH ->\u03B3 \u03B3 \u03C4 \u03C4 (x$10^6$)")

	fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize = (8,6), gridspec_kw = dict(height_ratios=[3,1]))
	plt.grid()

	hist_stack = [h_hh_ggbb_mc, h_vh_mc, h_ttgg_mc, h_zgamma_mc, h_wgamma_mc, h_ttg_mc, h_gg_mc, h_gjets_mc, h_dd_gjets_qcd_mc]
	hist_stack = sorted(hist_stack, key = lambda x:x.integral)

	h_data.plot(ax = ax1, alpha = 0.8, color = "black", errors=True, label = "Data")
	plot_stack(hist_stack, histtype = "stepfilled", ax = ax1, alpha = 0.8)
	h_tthh_ggbb.plot(ax=ax1, alpha = 0.8)
	h_tthh_ggWW.plot(ax=ax1, alpha = 0.8)
	h_tthh_ggTauTau.plot(ax=ax1, alpha = 0.8)

	ratio = h_data / h_mc
	ratio.plot(ax=ax2, errors=True, color="black")

	if log_y:
		ax1.set_yscale("log")

	if x_label is not None:
		ax2.set_xlabel(x_label)

	if y_label is not None:
		ax1.set_ylabel(y_label)

	if title is not None:
		ax1.set_title(title)

	if y_lim is not None:
		ax1.set_ylim(y_lim)

	if rat_lim is not None:
		ax2.set_ylim(rat_lim)

	if x_lim is not None:
		ax1.set_xlim(x_lim)

	if rat_label is not None:
		ax2.set_ylabel("Data/MC")

	ax2.set_ylim([0.0,2.0])
	plt.text(0,1, "CMS Preliminary", horizontalalignment = 'left', verticalalignment = 'bottom', transform = ax1.transAxes)
	plt.text(1,1, "137 fb$^{-1}$ (13 TeV)", horizontalalignment = 'right', verticalalignment = 'bottom', transform = ax1.transAxes)

	plt.savefig(linear_save_name)

# 4. Assign preselections that the plots will be based on
events_1lep_4jets_0tau = events[(events["n_leptons"] >= 1) & (events["n_jets"] >= 4) & (events["n_taus"] == 0)] #Semileptonic
events_0lep_5jets_0tau = events[(events["n_leptons"] == 0) & (events["n_taus"] == 0) & (events["n_jets"] >= 5)] #Hadronic 
events_2lep_0tau = events[(events["n_leptons"] == 2) & (events["n_taus"] == 0)] #Dileptonic
events_1lep1tau_0lep2tau = events[((events["n_leptons"] == 1) & (events["n_taus"] == 1)) | ((events["n_leptons"] == 0) & (events["n_taus"] == 2))] #Tau
events_3leptau = events[events["n_lep_tau"] >= 3] #Multilepton/Tau

# 5. Process the plot function based on data, background and signal
for events, presel_name in zip([events_1lep_4jets_0tau, events_0lep_5jets_0tau, events_2lep_0tau, events_1lep1tau_0lep2tau, events_3leptau], ["ttHH_semilep", "ttHH_hadronic", "ttHH_dileptonic", "ttHH_tau", "ttHH_multilepton_tau"]):
	
	#Data
	events_data = events[events["process_id"] == events_json["sample_id_map"]["Data"]]
	#MC
	events_mc = events[(events["process_id"] == events_json["sample_id_map"]["GJets_HT-100To200"]) + (events["process_id"] == events_json["sample_id_map"]["GJets_HT-200To400"]) + (events["process_id"] == events_json["sample_id_map"]["GJets_HT-400To600"]) + (events["process_id"] == events_json["sample_id_map"]["GJets_HT-40To100"]) + (events["process_id"] == events_json["sample_id_map"]["GJets_HT-600ToInf"]) + (events["process_id"] == events_json["sample_id_map"]["Diphoton"]) + (events["process_id"] == events_json["sample_id_map"]["TTGamma"]) + (events["process_id"] == events_json["sample_id_map"]["TTGG"]) + (events["process_id"] == events_json["sample_id_map"]["HH_ggbb"]) + (events["process_id"] == events_json["sample_id_map"]["VH_M125"]) + (events["process_id"] == events_json["sample_id_map"]["WGamma"]) + (events["process_id"] == events_json["sample_id_map"]["ZGamma"]) + (events["process_id"] == events_json["sample_id_map"]["DD_Description_GJetsQCD"])]
	events_gg_mc = events[events["process_id"] == events_json["sample_id_map"]["Diphoton"]]
	events_gjets_mc = events[(events["process_id"] == events_json["sample_id_map"]["GJets_HT-100To200"]) & (events["process_id"] == events_json["sample_id_map"]["GJets_HT-200To400"]) & (events["process_id"] == events_json["sample_id_map"]["GJets_HT-400To600"]) & (events["process_id"] == events_json["sample_id_map"]["GJets_HT-40To100"]) & (events["process_id"] == events_json["sample_id_map"]["GJets_HT-600ToInf"])]
	events_hh_ggbb_mc = events[events["process_id"] == events_json["sample_id_map"]["HH_ggbb"]]
	events_ttgg_mc = events[events["process_id"] == events_json["sample_id_map"]["TTGG"]]
	events_ttg_mc = events[events["process_id"] == events_json["sample_id_map"]["TTGamma"]]
	events_vh_mc = events[events["process_id"] == events_json["sample_id_map"]["VH_M125"]]
	events_wgamma_mc = events[events["process_id"] == events_json["sample_id_map"]["WGamma"]]
	events_zgamma_mc = events[events["process_id"] == events_json["sample_id_map"]["ZGamma"]]
	events_dd_gjets_qcd_mc = events[events["process_id"] == events_json["sample_id_map"]["DD_Description_GJetsQCD"]]
	#Signal
	events_tthh = events[(events["process_id"] == events_json["sample_id_map"]["ttHH_ggbb"]) & (events["process_id"] == events_json["sample_id_map"]["ttHH_ggWW"]) & (events["process_id"] == events_json["sample_id_map"]["ttHH_ggTauTau"])]
	events_tthh_ggbb = events[events["process_id"] == events_json["sample_id_map"]["ttHH_ggbb"]]
	events_tthh_ggWW = events[events["process_id"] == events_json["sample_id_map"]["ttHH_ggWW"]]
	events_tthh_ggTauTau = events[events["process_id"] == events_json["sample_id_map"]["ttHH_ggTauTau"]]

	#plots
	plots = {
		"Diphoton_mass" : {"bins" : "16,100,180", "x_label" : r"$m_{\gamma \gamma}$ [GeV]"},
		"n_jets" : {"bins" : "9, 1.5,10.5", "x_label" : r"$N_{\mathrm{jets}}$"},
		#Photon Kinematics 
		"LeadPhoton_pt_mgg" : { "bins": "16,-1,16", "x_label" : r"$p_{T}^{\gamma \gamma}/m_{\gamma \gamma}$"},
		"SubleadPhoton_pt_mgg" : {"bins" : "16,-1,7", "x_label" : r"$p_{T}^{\gamma \gamma}/m_{\gamma \gamma}$"},
		"LeadPhoton_eta" : {"bins" : "16,-3,3", "x_label" : "eta"},
		"SubleadPhoton_eta" : {"bins":"16,-3,3", "x_label":"eta"},
		"LeadPhoton_mvaID": {"bins":"40,-1,1","x_label":"photon ID MVA"},
		"SubleadPhoton_mvaID": {"bins":"40,-1,1", "x_label":"photon ID MVA"},
		#Diphoton Kinematics
		"Diphoton_pt_mgg" : { "bins" : "16,-1.5,17", "x_label": r"$p_{T}^{\gamma \gamma}/m_{\gamma \gamma}$"},
		"Diphoton_dR" : {"bins" : "16,-1,6", "x_label" : "Delta R"},
		#Highest and Second Highest b-jet scores
		"jet_1_btagDeepFlavB" : {"bins":"16,0,1", "x_label":"b-tag scores"}
	}

	for column, plot_info in plots.items(): 
		#Data
		data = events_data[column]
		#MC
		mc = events_mc[column]
		gg_mc = events_gg_mc[column]
		gjets_mc = events_gjets_mc[column]
		hh_ggbb_mc = events_hh_ggbb_mc[column]
		ttgg_mc = events_ttgg_mc[column]
		ttg_mc = events_ttg_mc[column]
		vh_mc = events_vh_mc[column]
		wgamma_mc = events_wgamma_mc[column]
		zgamma_mc = events_zgamma_mc[column]
		dd_gjets_qcd_mc = events_dd_gjets_qcd_mc[column]
		#Signal
		signal = events_tthh[column]
		tthh_ggbb = events_tthh_ggbb[column]
		tthh_ggWW = events_tthh_ggWW[column]
		tthh_ggTauTau = events_tthh_ggTauTau[column]

		plot_info["log_save_name"] = "/home/users/kmartine/public_html/plots/Spring_2022/log_%s_%s_dataMC.pdf" %(presel_name, column)

		make_log_ratio_plot(
			#Data
			data = data, 
			#MC
			mc = mc,
			mc_weight = events_mc["weight_central"],
			gg_mc = gg_mc,
			gg_mc_weight = events_gg_mc["weight_central"],
			gjets_mc = gjets_mc,
			gjets_mc_weight = events_gjets_mc["weight_central"],
			hh_ggbb_mc = hh_ggbb_mc,
			hh_ggbb_mc_weight = events_hh_ggbb_mc["weight_central"],
			ttgg_mc = ttgg_mc,
			ttgg_mc_weight = events_ttgg_mc["weight_central"],
			ttg_mc = ttg_mc,
			ttg_mc_weight = events_ttg_mc["weight_central"],
			vh_mc = vh_mc, 
			vh_mc_weight = events_vh_mc["weight_central"], 
			wgamma_mc = wgamma_mc, 
			wgamma_mc_weight = events_wgamma_mc["weight_central"], 
			zgamma_mc = zgamma_mc, 
			zgamma_mc_weight = events_zgamma_mc["weight_central"], 
			dd_gjets_qcd_mc = dd_gjets_qcd_mc,
			dd_gjets_qcd_mc_weight = events_dd_gjets_qcd_mc["weight_central"],
			#Signal
			signal = signal, 
			signal_weight = events_tthh["weight_central"],
			tthh_ggbb = tthh_ggbb, 
			tthh_ggbb_weight = events_tthh_ggbb["weight_central"]*1000, 
			tthh_ggWW = tthh_ggWW, 
			tthh_ggWW_weight = events_tthh_ggWW["weight_central"]*1000, 
			tthh_ggTauTau = tthh_ggTauTau,
			tthh_ggTauTau_weight = events_tthh_ggTauTau["weight_central"]*1000,
			**plot_info
		)

		plot_info["linear_save_name"] = "/home/users/kmartine/public_html/plots/Spring_2022/linear_%s_%s_dataMC.pdf" %(presel_name, column)

		make_linear_ratio_plot(
			#Data
			data = data,
			#MC
			mc = mc,
			mc_weight = events_mc["weight_central"],
			gg_mc = gg_mc,
			gg_mc_weight = events_gg_mc["weight_central"],
			gjets_mc = gjets_mc,
			gjets_mc_weight = events_gjets_mc["weight_central"],
			hh_ggbb_mc = hh_ggbb_mc,
			hh_ggbb_mc_weight = events_hh_ggbb_mc["weight_central"],
			ttgg_mc = ttgg_mc,
			ttgg_mc_weight = events_ttgg_mc["weight_central"],
			ttg_mc = ttg_mc,
			ttg_mc_weight = events_ttg_mc["weight_central"],
			vh_mc = vh_mc,
			vh_mc_weight = events_vh_mc["weight_central"],
			wgamma_mc = wgamma_mc,
			wgamma_mc_weight = events_wgamma_mc["weight_central"],
			zgamma_mc = zgamma_mc,
			zgamma_mc_weight = events_zgamma_mc["weight_central"],
			dd_gjets_qcd_mc = dd_gjets_qcd_mc,
			dd_gjets_qcd_mc_weight = events_dd_gjets_qcd_mc["weight_central"],
			#Signal
			signal = signal,
			signal_weight = events_tthh["weight_central"],
			tthh_ggbb = tthh_ggbb,
			tthh_ggbb_weight = events_tthh_ggbb["weight_central"]*100000,
			tthh_ggWW = tthh_ggWW,
			tthh_ggWW_weight = events_tthh_ggWW["weight_central"]*100000,
			tthh_ggTauTau = tthh_ggTauTau,
			tthh_ggTauTau_weight = events_tthh_ggTauTau["weight_central"]*1000000,
			**plot_info
		)




