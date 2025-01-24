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

tech_names = {
    "CH4 (g) pipeline": "natural gas pipeline",
    "CH4 (g) submarine pipeline": "natural gas pipeline (submarine)",
    "H2 (g) pipeline": "hydrogen pipeline",
    "H2 (g) pipeline repurposed": "hydrogen pipeline (repurposed)",
    "H2 (g) submarine pipeline": "hydrogen pipeline (submarine)",
    "H2 liquefaction": "hydrogen liquefaction",
    "HVAC overhead": "HVAC transmission line (overhead)",
    "HVDC overhead": "HVDC transmission line (overhead)",
    "HVDC submarine": "HVDC transmission line (submarine)",
    "PHS": "pumped hydro storage",
    "SMR": "steam methane reforming",
    "SMR CC": "steam methane reforming with carbon capture",
    "biomass CHP": "CHP (biomass)",
    "biomass CHP capture": "CHP (biomass with carbon capture)",
    "central air-sourced heat pump": "heat pump (air-sourced, central)",
    "central gas CHP": "CHP (gas, central)",
    "central gas boiler": "gas boiler (central)",
    "central ground-sourced heat pump": "heat pump (ground-sourced, central)",
    "central resistive heater": "resistive heater (central)",
    "central solar thermal": "solar thermal (central)",
    "central solid biomass CHP": "CHP (solid biomass, central)",
    "central water tank storage": "thermal storage (water tank, central)",
    "decentral CHP": "CHP (decentral)",
    "decentral air-sourced heat pump": "heat pump (air-sourced, decentral)",
    "decentral gas boiler": "gas boiler (decentral)",
    "decentral ground-sourced heat pump": "heat pump (ground-sourced, decentral)",
    "decentral resistive heater": "resistive heater (decentral)",
    "decentral solar thermal": "solar thermal (decentral)",
    "decentral water tank storage": "thermal storage (water tank, decentral)",
    "direct air capture": "direct air capture (DAC)",
    "gas": "fossil gas",
    "helmeth": "HELMETH (direct power-to-methane)",
    "hydro": "reservoir hydro",
    "hydrogen storage tank incl. compressor": "hydrogen storage (steel tank)",
    "hydrogen storage underground": "hydrogen storage (underground)",
    "offwind": "offshore wind",
    "offwind-ac-connection-submarine": "AC grid connection (submarine)",
    "offwind-ac-connection-underground": "AC grid connection (underground)",
    "offwind-ac-station": "AC grid connection (station)",
    "offwind-dc-connection-submarine": "DC grid connection (submarine)",
    "offwind-dc-connection-underground": "DC grid connection (underground)",
    "offwind-dc-station": "DC grid connection (station)",
    "oil": "fossil oil",
    "onwind": "onshore wind",
    "ror": "run of river",
    "solar": "solar PV",
    "solar-rooftop": "solar PV (rooftop)",
    "solar-utility": "solar PV (utility-scale)",
}

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


def prepare_energy_balances(networks_dict, country):
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )

    energy_balance = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():

        n = pypsa.Network(filename)

        if country == "EU":
            df_annual = n.statistics.energy_balance()
        else:
            df_annual = n.statistics.energy_balance(groupby=["carrier", "bus_carrier", "country"])
            idx = pd.IndexSlice
            df_annual = df_annual.loc[idx[:,:,:,country]]

        df_annual = df_annual.reorder_levels(['bus_carrier', 'component', 'carrier'])
        df_annual = df_annual.sort_index()

        energy_balance = energy_balance.reindex(df_annual.index.union(energy_balance.index))
        energy_balance.loc[df_annual.index, label] = df_annual

    energy_balance = energy_balance.fillna(0)

    return energy_balance


def to_csv(df, run_name, country):
    df.to_csv(f"workflow/results/{run_name}/csvs/{country}_energy_balances.csv")


def plot_balances(run_name, config, country, energy_threshold):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        f"workflow/results/{run_name}/csvs/{country}_energy_balances.csv", index_col=list(range(3)), header=list(range(4))
    )

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [
        i for i in balances_df.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():
        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh
        df = df / 1e6

        # remove trailing link ports
        df.index = [
            (
                i[:-1]
                if (
                    (i not in ["co2", "NH3", "H2"])
                    and (i[-1:] in ["0", "1", "2", "3", "4"])
                )
                else i
            )
            for i in df.index
        ]

        df = df.groupby(df.index.map(rename_techs)).sum()

        to_drop = df.index[
            df.abs().max(axis=1) < energy_threshold
        ]

        units = "MtCO2/a" if v[0] in co2_carriers else "TWh/a"

        df = df.drop(to_drop)

        if df.empty:
            continue

        new_index = preferred_order.intersection(df.index).append(
            df.index.difference(preferred_order)
        )

        new_columns = df.columns.sort_values()

        fig, ax = plt.subplots(figsize=(12, 8))

        df.loc[new_index, new_columns].T.plot(
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

        if v[0] in co2_carriers:
            ax.set_ylabel("CO2 [MtCO2/a]")
        else:
            ax.set_ylabel("Energy [TWh/a]")

        ax.set_xlabel("")

        ax.grid(axis="x")

        ax.legend(
            handles,
            labels,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=[1, 1],
            frameon=False,
        )

        fig.savefig(f"workflow/results/{run_name}/{country}_balances/" + k + ".svg", bbox_inches="tight")
        plt.close(fig)

        
run_name = "myopic-default-2025-2050-5-T-H-B-I-A"
config = "config.myopic_main.yaml" 
country = "EU" # EU means all European countries
energy_threshold = 5 # in TWh; different between DE and EU

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

energy_balance = prepare_energy_balances(networks_dict, country)

to_csv(energy_balance, run_name, country)

plot_balances(run_name, config, country)
