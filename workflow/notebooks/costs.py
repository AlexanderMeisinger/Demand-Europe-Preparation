import yaml
import pypsa
import warnings
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime
from cartopy import crs as ccrs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
import os
import xarray as xr
import cartopy
import sys

PATH = "../pypsa-eur/"
sys.path.append(os.path.join(PATH, "scripts/"))
from _helpers import rename_techs

#plt.style.use(["bmh", "matplotlibrc"])
xr.set_options(display_style="html")


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)

nice_names = {
        "unsustainable solid biomass": "Unsustainable Solid Biomass",
        "unsustainable bioliquids": "Unsustainable Liquids",
        "solid biomass for industry CC": "Solid Biomass for Industry CC",
        "solid biomass for industry": "Solid Biomass for Industry",
        "process emissions CC": "Process Emissions CC",
        "oil primary": "Oil Primary",
        "oil boiler": "Oil Boiler",
        "nuclear": "Nuclear Energy",
        "methanolisation": "Methanolisation",
        "lignite": "Lignite",
        "gas for industry CC": "Gas for Industry CC",
        "electricity distribution grid": "Electricity Distribution Grid",
        "coal": "Coal",
        "co2 sequestered": "CO2 Sequestered",
        "biomass boiler": "Biomass Boiler",
        "SMR CC": "SMR CC",
        "SMR": "SMR",
        "H2 pipeline": "H2 Pipeline",
        "H2 Store": "H2 Store",
        "H2 Electrolysis": "H2 Electrolysis",
        "Fischer-Tropsch": "Fischer-Tropsch",
        "DAC": "DAC",
        "CCGT": "CCGT",
        "hot water storage": "Hot Water Storage",
        "methanation": "Methanation",
        "gas": "Natural Gas",
        "gas boiler": "Gas Boiler",
        "CHP": "CHP",
        "resistive heater": "Resistive Heater",
        "air heat pump": "Air Heat Pump",
        "ground heat pump": "Ground Heat Pump",
        "solar rooftop": "Solar Rooftop",
        "solar PV": "Solar PV",
        "offshore wind (AC)": "Offshore Wind (AC)",
        "onshore wind": "Onshore Wind",
        "biogas": "Biogas",
        "solid biomass": "Solid Biomass",
        "hydroelectricity": "Hydroelectricity",
        "transmission lines": "Transmission Lines",
        "solar-hsat": "Solar-HSAT",
        "shipping oil": "Shippping Oil",
        "shipping methanol": "Shipping Methanol", 
        "oil refining": "Oil Refining",
        "naphtha for industry": "Naphtha for Industry",
        "land transport oil": "Land Transport Oil",
        "kerosene for aviation": "Kerosene for Aviation",
        "gas for industry": "Gas for Industry",
        "coal for industry": "Coal for Industry",
        "co2": "CO2",
        "agriculture machinery oil": "Agriculture Machinery Oil",
        "low-temperature heat for industry": "Low-temperature Heat for Industry",
        "land transport EV": "Land Transport EV",
        "industry methanol": "Industry Methanol", 
        "industry electricity": "Industry Electicity",
        "heat": "Heat", 
        "electricity": "Electricity", 
        "agriculture heat": "Agriculture Heat", 
        "agriculture electricity": "Agriculture Electricity"
    }

def prepare_costs(networks_dict, country):
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )

    costs = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():

        n = pypsa.Network(filename)

        if country == "EU":
            df_capital_annual = pd.concat([n.statistics.capex()], keys=["capital"])
        else:
            df_capital_annual = pd.concat([n.statistics.capex(groupby=["carrier", "bus_carrier", "country"])], keys=["capital"])
            idx = pd.IndexSlice
            df_capital_annual = df_capital_annual.loc[idx[:,:,:,:,country]]
            df_capital_annual = df_capital_annual.reorder_levels([None, 'bus_carrier', 'component', 'carrier'])
            df_capital_annual = df_capital_annual.sort_index()

        costs = costs.reindex(df_capital_annual.index.union(costs.index))
        costs.loc[df_capital_annual.index, label] = df_capital_annual

        if country == "EU":
            df_marginal_costs_annual = pd.concat([n.statistics.opex()], keys=["marginal"])
        else:
            df_marginal_costs_annual = pd.concat([n.statistics.opex(groupby=["carrier", "bus_carrier", "country"])], keys=["marginal"])
            idx = pd.IndexSlice
            df_marginal_costs_annual = df_marginal_costs_annual.loc[idx[:,:,:,:,country]]
            df_marginal_costs_annual = df_marginal_costs_annual.reorder_levels([None, 'bus_carrier', 'component', 'carrier'])
            df_marginal_costs_annual = df_marginal_costs_annual.sort_index()

        costs = costs.reindex(df_marginal_costs_annual.index.union(costs.index))
        costs.loc[df_marginal_costs_annual.index, label] = df_marginal_costs_annual

    costs = costs.fillna(0)

    return costs


def to_csv(df, run_name, country):
    df.to_csv(f"workflow/results/{run_name}/csvs/{country}_costs.csv")


def plot_costs(run_name, config, country, cost_threshold):
    if country == "EU":
        cost_df = pd.read_csv(
            f"workflow/results/{run_name}/csvs/{country}_costs.csv", index_col=list(range(3)), header=list(range(4))
        )
        df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()
    else:
        cost_df = pd.read_csv(
            f"workflow/results/{run_name}/csvs/{country}_costs.csv", index_col=list(range(4)), header=list(range(4))
        )
        df = cost_df.groupby(cost_df.index.get_level_values(3)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < cost_threshold]

    df = df.drop(to_drop)

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    # new_columns = df.sum().sort_values().index

    fig, ax = plt.subplots(figsize=(12, 8))

    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[config["plotting"]["tech_colors"][i] for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()

    # Map the labels to their nice names
    labels = [nice_names.get(label, label) for label in labels]
    
    handles.reverse()
    labels.reverse()

    #ax.set_ylim([0, config["plotting"]["costs_max"]])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False
    )

    fig.savefig(f"workflow/results/{run_name}/{country}_balances/{country}_costs2.svg", bbox_inches="tight")
    plt.close(fig)

        
run_name = "myopic-default-2025-2050-5-T-H-B-I-A"
config = "config.myopic_main.yaml" 
country = "DE" # EU means all European countries
cost_threshold = 1 # in TWh; different between DE and EU

with open("/mnt/e/H2GMA/Github/Europe/analyse-h2g-a-ap3-eu/config/" + config) as file:
    config = yaml.safe_load(file)

networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/{run_name}"
        + f"/postnetworks/base_s_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in config["scenario"]["ll"]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }

costs = prepare_costs(networks_dict, country)

to_csv(costs, run_name, country)

plot_costs(run_name, config, country, cost_threshold)
