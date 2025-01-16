import pypsa
import yaml
import cartopy
import sys
import re
import os

import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from itertools import product
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

from vresutils.costdata import annuity

PATH = "/mnt/e/H2GMA/Github/Europe/pypsa-eur"

sys.path.append(os.path.join(PATH, "scripts/"))
from _helpers import rename_techs

#plt.style.use(["bmh", "matplotlibrc"])
xr.set_options(display_style="html")

CLUSTERS = 39

MAIN_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
IMP_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
SHP_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
COST_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
	
def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis"]:  # , "H2 liquefaction"]:
        return "power-to-hydrogen"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #    return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam methane reforming")
    elif "DAC" in tech:
        return "direct air capture"
    elif "CC" in tech or "sequestration" in tech:
        return "carbon capture"
    elif tech == "oil" or tech == "gas":
        return "fossil oil and gas"
    else:
        return tech
    
def rename_techs_carbon_balances(tech):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]
    for ptr in prefix_to_remove:
        if tech[: len(ptr)] == ptr:
            tech = tech[len(ptr) :]
    if tech == "biogas to gas":
        return "biogas upgrading"
    elif tech == "agriculture machinery oil emissions":
        return "agriculture machinery"
    elif tech == "shipping methanol emissions":
        return "shipping fuels"
    elif tech == "DAC":
        return "direct air capture"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam methane reforming")
    else:
        return tech
    
preferred_order = pd.Index(
    [
        "transmission lines",
        "electricity distribution grid",
        "fossil oil and gas",
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
        "helmeth",
        "methanation",
        "power-to-gas",
        "power-to-hydrogen",
        "H2 pipeline",
        "H2 liquefaction",
        "H2 storage",
        "hydrogen storage",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
        "CCS",
        "carbon capture and sequestration",
        "DAC",
        "direct air capture",
    ]
)

def parse_index(c):
    clusters = c[0]

    lv = c[1]

    match = re.search(r"onwind\+p([0-9.]*)", c[2])
    onw = 100.0 if match is None else 100 * float(match.groups()[0])

    #h2 = "no H2 grid" if "noH2network" in c[2] else "H2 grid"
    h2 = c[3]

    to_return = (clusters, lv, onw, h2)

    return to_return

	
def load_main(
    scenarios=None, clusters=None, rename=True, with_space=False
):
    if scenarios is None:
        scenarios = MAIN_SCENARIOS

    clusters = CLUSTERS

    #horizon = "2030" if "rev0" in scenarios else "2050"

    costs = pd.read_csv(
        scenarios + f"/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2]
    )

    #costs = costs.xs(horizon, level="planning_horizon", axis=1)

    #names = ["clusters", "lv", "onw", "planning_horizon"]
    names = ["clusters", "lv", "onw", "H2"]

    costs.columns = pd.MultiIndex.from_tuples(
        [parse_index(c) for c in costs.columns], names=names
    )

    if not with_space:
        costs = costs.xs(str(clusters), level="clusters", axis=1)

    df = costs.groupby(level=2).sum().div(1e9)

    to_drop = df.index[df.max(axis=1).fillna(0.0) < 1.2]
    print(to_drop)
    df.drop(to_drop, inplace=True)

    # H2G-A: Probably change neccessary
    if "-imp" in scenarios:
        # imports for methanol, kerosene and naphtha at 120 €/MWh
        print("add import costs")
        df.loc["green e-fuel imports"] = (1026.64 + 546.36) * 120e6 / 1e9  # bn€/a

    return df

def load_main_supply_energy(
    scenarios=None,
    clusters=None,
    rename=True,
    with_resolution=False,
    with_space=False,
    carrier="energy",
):
    if scenarios is None:
        scenarios = MAIN_SCENARIOS

    clusters = CLUSTERS

    #horizon = "2030" if "rev0" in scenarios else "2050"

    df = pd.read_csv(
        scenarios + "/csvs/supply_energy.csv", index_col=[0, 1, 2], header=[0, 1, 2, 3]
    )

    co2_carriers = ["co2", "co2 stored", "process emissions"]
    if carrier == "energy":
        carrier = [i for i in df.index.levels[0] if i not in co2_carriers]

    df = df.loc[carrier].groupby(level=2).sum().div(1e6)  # TWh / MtCO2
    df.index = [
        i[:-1]
        if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0", "1", "2", "3"]))
        else i
        for i in df.index
    ]

    #df = df.xs(horizon, level="planning_horizon", axis=1)

    names = ["clusters", "lv", "onw", "h2"]
    if with_resolution:
        names += ("res",)

    df.columns = pd.MultiIndex.from_tuples(
        [parse_index(c) for c in df.columns], names=names
    )

    if not with_space:
        df = df.xs(str(clusters), level="clusters", axis=1)

    if rename or callable(rename):
        func = rename if callable(rename) else rename_techs_tyndp
        df = df.groupby(df.index.map(func)).sum()

    to_drop = df.index[df.abs().max(axis=1).fillna(0.0) < 10]
    df.drop(to_drop, inplace=True)

    order = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )
    df = df.loc[order]

    if "-imp" in scenarios and carrier == "energy":
        # imports for methanol, kerosene and naphtha
        df.loc["green e-fuel imports"] = 1026.64 + 546.36  # TWh

    return df

def carbon_balances(scenarios, onw=100):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        scenarios + "/csvs/supply_energy.csv", index_col=[0, 1, 2], header=[0, 1, 2, 3]
    )

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [
        i for i in balances_df.index.levels[0] if i not in co2_carriers
    ]
    balances["carbon"] = [i for i in balances_df.index.levels[0] if i in co2_carriers]

    key = "co2"

    df = balances_df.loc[balances[key]]

    df = df.groupby(level=2).sum().div(1e6)

    df.index = [
        i[:-1]
        if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0", "1", "2", "3"]))
        else i
        for i in df.index
    ]

    df = df.groupby(rename_techs_carbon_balances).sum()

    df.columns = pd.MultiIndex.from_tuples(
        [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "h2"]
    )

    df = df.xs((onw, str(39)), level=["onw", "clusters"], axis=1)

    df.drop("co2", inplace=True)

    order = pd.Index(
        [
            "liquid hydrocarbons emissions",
            "methanol emissions",
            "process emissions CC",
            "gas for industry CC",
            "gas CHP CC",
            "gas CHP",
            "OCGT",
            "gas boiler",
            "steam methane reforming",
            "steam methane reforming CC",
            "biogas upgrading",
            "solid biomass CHP CC",
            "solid biomass for industry CC",
            "direct air capture",
        ]
    )

    order = order.intersection(df.index).append(df.index.difference(order))
    df = df.loc[order]

    df = df.loc[df.abs().max(axis=1) > 0.01]

    return df

    
SCENARIOS = {
    (0, 0, 0, 0): (MAIN_SCENARIOS, 100), 
    (1, 0, 0, 0): (COST_SCENARIOS, 100),
    (0, 1, 0, 0): (IMP_SCENARIOS, 100),
    (0, 0, 1, 0): (SHP_SCENARIOS, 100),
    (0, 0, 0, 1): (MAIN_SCENARIOS, 100),
}

NAMES = ["optimistic_costs", "imports", "hydrogen_in_shipping", "no_onwind"]

tsc = pd.concat(
    {
        k: load_main(scenarios).xs(onw, level="onw", axis=1)
        for k, (scenarios, onw) in SCENARIOS.items()
    },
    names=NAMES,
)
tsc = tsc.xs("vopt", level="lv", axis=1)

tsc.index.names = tsc.index.names[:-1] + ["carrier"]
tsc = tsc.stack([0]).to_xarray()
tsc.name = "costs"

energy = pd.concat(
    {
        k: load_main_supply_energy(scenarios).xs(onw, level="onw", axis=1)
        for k, (scenarios, onw) in SCENARIOS.items()
    },
    names=NAMES,
)

energy = energy.xs("vopt", level="lv", axis=1)
energy.index.names = energy.index.names[:-1] + ["carrier"]
energy = energy.stack([0]).to_xarray()
energy.name = "energy"

co2 = pd.concat(
    {k: carbon_balances(scenarios, onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)

co2.index.names = co2.index.names[:-1] + ["carrier"]
co2 = co2.stack([0, 1]).to_xarray()
co2.name = "co2"

ds = xr.merge([tsc, energy, co2]).round(2)
comp = dict(zlib=True, complevel=9)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("scenarios_myopic.nc", encoding=encoding)

