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

PATH = "../../../pypsa-eur/"
sys.path.append(os.path.join(PATH, "scripts/"))
from _helpers import rename_techs

plt.style.use(["bmh", "matplotlibrc"])
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


def calculate_costs(n, label, costs):
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        capital_costs = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        marginal_costs = p * c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped

    # add back in all hydro
    # costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
    # costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
    # costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

    return costs

def plot_costs(run_name, config, n_header):
    cost_df = pd.read_csv(
        f"../../../pypsa-eur/results/myopic/{run_name}/csvs/costs.csv", index_col=list(range(3)), header=list(range(n_header))
    )

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < config["plotting"]["costs_threshold"]]

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

    ax.set_ylim([0, config["plotting"]["costs_max"]])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False
    )

    fig.savefig(f"../results/{run_name}/costs/costs.svg", bbox_inches="tight")
    plt.close(fig)


run_name = "myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
config = "config.pathways-myopics_default_cb_red.yaml" 


with open("/mnt/e/H2GMA/Github/Europe/analyse-h2g-a-ap3-eu/config/" + config) as file:
    config = yaml.safe_load(file)

plot_costs(run_name, config, n_header=4)