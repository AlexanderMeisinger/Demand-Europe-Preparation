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

xr.set_options(display_style="html")


CLUSTERS = 39

MAIN_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
IMP_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
SHP_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"
COST_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A-co2-budget"

with open("/mnt/e/H2GMA/Github/Europe/analyse-h2g-a-ap3-eu/config/config.pathways-myopics_default_cb_red.yaml") as file:
    config = yaml.safe_load(file)

#ToDo: Check and adapt tech_colors
tech_colors = config["plotting"]["tech_colors"]
tech_colors["battery electric vehicles"] = tech_colors["BEV charger"]
tech_colors["other"] = "#454545"
tech_colors["building heat demand"] = tech_colors["heat"]
tech_colors["ambient heat"] = tech_colors["heat pump"]
tech_colors["residential electricity demand"] = "#72709c"
tech_colors["industry electricity demand"] = tech_colors["electricity"]
tech_colors["hydrogen demand"] = tech_colors["land transport fuel cell"]
tech_colors["agriculture machinery"] = tech_colors["land transport fuel cell"]
#tech_colors["methane demand"] = tech_colors["helmeth"] #ToDo: Adapt
tech_colors["liquid hydrocarbon demand"] = tech_colors["kerosene for aviation"]
tech_colors["aviation fuels"] = tech_colors["kerosene for aviation"]
tech_colors["shipping fuels"] = tech_colors["shipping methanol"]
tech_colors["biomass demand"] = tech_colors["biogas"]
tech_colors["biogas upgrading"] = tech_colors["biogas"]
tech_colors["hydrogen for industry"] = tech_colors["H2 for industry"]
tech_colors["hydrogen-to-power/heat"] = tech_colors["gas-to-power/heat"]
tech_colors["hydrogen for land transport"] = "#8487e8"


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
    

def rename_techs_balances(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech:
        return "ambient heat"
    elif tech in ["H2 Electrolysis"]:  # , "H2 liquefaction"]:
        return "power-to-hydrogen"
    elif "solar" in tech:
        return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif tech == "DAC":
        return "direct air capture"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech == "oil" or tech == "gas":
        return "fossil oil and gas"
    elif tech in ["BEV charger", "V2G", "Li ion", "land transport EV"]:
        return "battery electric vehicles"
    elif tech in ["biogas", "solid biomass"]:
        return "biomass"
    elif tech in ["electricity"]:
        return "residential electricity demand"
    elif tech in ["industry electricity", "agriculture electricity"]:
        return "industry electricity demand"
    elif tech in ["agriculture heat", "heat", "low-temperature heat for industry"]:
        return "heat demand"
    elif "solid biomass for industry" in tech:
        return "biomass demand"
    elif "gas for industry" in tech:
        return "methane demand"
    elif tech in ["H2 for industry", "land transport fuel cell"]:
        return "hydrogen demand"
    elif tech in [
        "kerosene for aviation",
        "naphtha for industry",
        "shipping methanol",
        "agriculture machinery oil",
    ]:
        return "liquid hydrocarbon demand"
    elif tech in [
        "transmission lines",
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "H2",
        "electricity distribution grid",
        "SMR",
        "SMR CC",
        "OCGT",
        "CHP",
        "gas boiler",
        "H2 Fuel Cell",
        "resistive heater",
        "battery storage",
        "methanation",
    ]:
        return "other"
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


def parse_index(c, with_resolution=False):
    clusters = c[0]

    lv = c[1]

    match = re.search(r"onwind\+p([0-9.]*)", c[2])
    onw = 100.0 if match is None else 100 * float(match.groups()[0])

    #h2 = "no H2 grid" if "noH2network" in c[2] else "H2 grid"
    h2 = c[3]

    to_return = (clusters, lv, onw, h2)

    if with_resolution:
        match = re.findall(r"(\d+)H", c[2])
        to_return += (int(match[0]),)

    return to_return


def load_main(
    scenarios=None, clusters=None, rename=True, with_resolution=False, with_space=False
):
    if scenarios is None:
        scenarios = MAIN_SCENARIOS

    clusters = CLUSTERS

    #horizon = "2030" if "rev0" in scenarios else "2050"

    costs = pd.read_csv(
        scenarios + f"/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2]
    )

    #costs = costs.xs(horizon, level="planning_horizon", axis=1)

    names = ["clusters", "lv", "onw", "h2"]
    if with_resolution:
        names += ("res",)

    costs.columns = pd.MultiIndex.from_tuples(
        [parse_index(c, with_resolution) for c in costs.columns], names=names
    )

    if not with_space:
        costs = costs.xs(str(clusters), level="clusters", axis=1)

    df = costs.groupby(level=2).sum().div(1e9)

    if rename:
        df = df.groupby(df.index.map(rename_techs_tyndp)).sum()

    to_drop = df.index[df.max(axis=1).fillna(0.0) < 1.2]
    print(to_drop)
    df.drop(to_drop, inplace=True)

    order = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )
    df = df.loc[order]

    # H2G-A: Probably change neccessary
    if "-imp" in scenarios:
        # imports for methanol, kerosene and naphtha at 120 €/MWh
        print("add import costs")
        df.loc["green e-fuel imports"] = (1026.64 + 546.36) * 120e6 / 1e9  # bn€/a
        tech_colors["green e-fuel imports"] = "#46caf0"

    return df


def load_main_capacities(
    scenarios=None,
    clusters=None,
    rename=True,
    with_resolution=False,
    with_space=False,
    merge=True,
):
    if scenarios is None:
        scenarios = MAIN_SCENARIOS

    clusters = CLUSTERS

    #horizon = "2030" if "rev0" in scenarios else "2050"

    df = pd.read_csv(
        scenarios + f"/csvs/capacities.csv", header=[0, 1, 2, 3], index_col=[0, 1]
    )

    #df = df.xs(horizon, level="planning_horizon", axis=1)

    names = ["clusters", "lv", "onw", "h2"]
    if with_resolution:
        names += ("res",)

    df.columns = pd.MultiIndex.from_tuples(
        [parse_index(c, with_resolution) for c in df.columns], names=names
    )

    if not with_space:
        df = df.xs(str(clusters), level="clusters", axis=1)

    if rename:
        grouper = [
            df.index.get_level_values(0),
            df.index.get_level_values(1).map(rename_techs_tyndp),
        ]
        df = df.groupby(grouper).sum()

    to_drop = df.index[df.max(axis=1).fillna(0.0) < 10]
    df.drop(to_drop, inplace=True)

    twh = df.xs("stores", level=0).div(1e6)  # TWh

    to_drop = [
        "CCS",
        "biogas",
        "co2",
        "fossil oil and gas",
        "solid biomass",
    ]
    twh.drop(twh.index.intersection(to_drop), inplace=True)

    gw = df.drop(["stores", "lines"]).div(1e3)  # GW

    if merge:
        gw = gw.groupby(level=1).sum()
        techs = gw.index
        kwargs = dict()
    else:
        techs = gw.index.levels[1]
        kwargs = dict(level=1)

    to_drop = [
        "fossil oil and gas",
        "transmission lines",
        "DAC",
        "direct air capture",
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "CCS",
        "carbon capture" "biogas",
        "gas for industry",
        "hot water storage",
        "solid biomass for industry",
        "process emissions",
    ]
    gw.drop(techs.intersection(to_drop), **kwargs, inplace=True)

    return gw, twh


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
        [parse_index(c, with_resolution) for c in df.columns], names=names
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
        tech_colors["green e-fuel imports"] = "#46caf0"

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

    df = df.xs((onw, str(CLUSTERS)), level=["onw", "clusters"], axis=1)

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


def rename_techs_h2_balances(tech):
    if tech == "H2 for industry":
        return "hydrogen for industry"
    elif tech == "Sabatier":
        return "methanation"
    elif tech == "H2 Electrolysis":
        return "power-to-hydrogen"
    elif tech == "land transport fuel cell":
        return "hydrogen for land transport"
    elif tech == "H2 Fuel Cell":
        return "hydrogen-to-power/heat"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam methane reforming")
    else:
        return tech


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
tsc.index.names = tsc.index.names[:-1] + ["carrier"]
tsc = tsc.stack([0, 1]).to_xarray()
tsc.name = "costs"
energy = pd.concat(
    {
        k: load_main_supply_energy(scenarios).xs(onw, level="onw", axis=1)
        for k, (scenarios, onw) in SCENARIOS.items()
    },
    names=NAMES,
)
energy.index.names = energy.index.names[:-1] + ["carrier"]
energy = energy.stack([0, 1]).to_xarray()
energy.name = "energy"
co2 = pd.concat(
    {k: carbon_balances(scenarios, onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)
co2.index.names = co2.index.names[:-1] + ["carrier"]
co2 = co2.stack([0, 1]).to_xarray()
co2.name = "co2"

def read_capacities(scenarios, onw):
    gw, twh = load_main_capacities(scenarios, merge=False)

    gw = gw.xs(onw, level="onw", axis=1)
    twh = twh.xs(onw, level="onw", axis=1)

    gen = gw.loc[["generators", "storage_units"]].groupby(level=1).sum()

    con = gw.loc[["links"]].groupby(level=1).sum()
    return gen, con, twh

cap = pd.concat(
    {
        k: pd.concat(
            read_capacities(scenarios, onw),
            keys=["generation", "conversion", "storage"],
        )
        for k, (scenarios, onw) in SCENARIOS.items()
    },
    names=NAMES,
)

cap.index.names = cap.index.names[:-2] + ["category", "carrier"]
cap = cap.stack([0, 1]).unstack("category").to_xarray()
ds = xr.merge([tsc, energy, co2, cap]).round(2)
comp = dict(zlib=True, complevel=9)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("scenarios.nc", encoding=encoding)