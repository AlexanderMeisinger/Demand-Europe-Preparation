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
NOH2GRID_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-noh2grid-2025-2050-5-T-H-B-I-A"
LOWCARBON_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-low_carbon_budget-2025-2050-5-T-H-B-I-A"

with open("/mnt/e/H2GMA/Github/Europe/analyse-h2g-a-ap3-eu/config/config.myopic_main.yaml") as file:
    config = yaml.safe_load(file)


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


def energy_balances(scenarios, sector, onw=100):
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = pd.read_csv(
        scenarios + "/csvs/supply_energy.csv", index_col=[0, 1, 2], header=[0, 1, 2, 3]
    )

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [
        i for i in balances_df.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():
        if k == sector:
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
                df.abs().max(axis=1) < config["plotting"]["energy_threshold"] / 10
            ]

            df = df.drop(to_drop)

            if df.empty:
                continue

            df.columns = pd.MultiIndex.from_tuples(
                [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "h2"]
            )

            df = df.xs((onw, str(CLUSTERS)), level=["onw", "clusters"], axis=1)
            
            new_index = preferred_order.intersection(df.index).append(
                        df.index.difference(preferred_order)
                    )

            new_columns = df.columns.sort_values()

            df = df.loc[new_index, new_columns]

    return df 


def costs(scenarios, onw=100):
    cost_df = pd.read_csv(
        scenarios + "/csvs/costs.csv", index_col=list(range(3)), header=[0, 1, 2, 3]
    )

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < config["plotting"]["costs_threshold"]]

    df = df.drop(to_drop)

    df.columns = pd.MultiIndex.from_tuples(
                [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "h2"]
            )

    df = df.xs((onw, str(CLUSTERS)), level=["onw", "clusters"], axis=1)

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    df = df.loc[new_index]
    
    return df


SCENARIOS = {
    (0, 0): (MAIN_SCENARIOS, 100), 
    (1, 0): (LOWCARBON_SCENARIOS, 100),
    #(0, 1, 0, 0): (IMP_SCENARIOS, 100),
    (0, 1): (NOH2GRID_SCENARIOS, 100),
    #(0, 0, 0, 1): (MAIN_SCENARIOS, 100),
}

NAMES = ["low_carbon", "no_h2grid"]

cost = pd.concat(
    {k: costs(scenarios, onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)
cost.index.names = cost.index.names[:-1] + ["carrier"]
cost = cost.stack([0, 1]).to_xarray()
cost.name = "costs"


energy = pd.concat(
    {k: energy_balances(scenarios, "energy", onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)

energy.index.names = energy.index.names[:-1] + ["carrier"]
energy = energy.stack([0, 1]).to_xarray()
energy.name = "energy"

co2 = pd.concat(
    {k: energy_balances(scenarios, "co2", onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)

co2.index.names = co2.index.names[:-1] + ["carrier"]
co2 = co2.stack([0, 1]).to_xarray()
co2.name = "co2"

h2 = pd.concat(
    {k: energy_balances(scenarios, "H2", onw) for k, (scenarios, onw) in SCENARIOS.items()},
    names=NAMES,
)

h2.index.names = h2.index.names[:-1] + ["carrier"]
h2 = h2.stack([0, 1]).to_xarray()
h2.name = "hydrogen"


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
ds = xr.merge([cost, energy, co2, h2, cap]).round(2)
comp = dict(zlib=True, complevel=9)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("scenarios-h2-co2-energy-costs.nc", encoding=encoding)