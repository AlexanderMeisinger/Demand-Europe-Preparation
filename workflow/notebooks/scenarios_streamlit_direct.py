# This script prepares results from PyPSA-Eur for Streamlit visualization
# Author: Alexander Meisinger
# Base: https://doi.org/10.1016/j.joule.2023.06.016 and https://github.com/PyPSA/pypsa-eur

import os
import re
import sys
import yaml
import pandas as pd
import numpy as np
import xarray as xr
import pypsa

# Set the main path for script imports and configuration
PATH = "/mnt/e/H2GMA/Github/Europe/pypsa-eur"
sys.path.append(os.path.join(PATH, "scripts/"))
from _helpers import rename_techs

# Configure global settings
xr.set_options(display_style="html")


# Helper function to parse scenario index
# Add information about cluster level, grid- and wind expansion, time horizon as headline of final dataset
def parse_index(c):
    clusters = c[0] 
    lv = c[1]
    match = re.search(r"onwind\+p([0-9.]*)", c[2])
    onw = 100.0 if match is None else 100 * float(match.groups()[0])
    year = c[3]
    to_return = (clusters, lv, onw, year)

    return to_return


# Process capacity data
def capacity(scenarios, country, onw=100):
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/postnetworks/base_s_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in config["scenario"]["ll"]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }
    
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )

    cap = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():

        n = pypsa.Network(filename)
    
        if country == "EU":
            df_cap_annual = pd.concat([n.statistics.optimal_capacity()])
        else:
            df_cap_annual = pd.concat([n.statistics.optimal_capacity(groupby=["carrier", "country"])])
            idx = pd.IndexSlice
            df_cap_annual = df_cap_annual.loc[idx[:,:,country]]
            
            df_cap_annual = df_cap_annual.reorder_levels(['component', 'carrier'])
            df_cap_annual = df_cap_annual.sort_index()

        cap = cap.reindex(df_cap_annual.index.union(cap.index))
        cap.loc[df_cap_annual.index, label] = df_cap_annual

    df = cap.fillna(0)

    #df = df.groupby(df.index.map(rename_techs)).sum()
    
    df.columns = pd.MultiIndex.from_tuples(
                [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "year"]
            )

    df = df.xs((onw, config["scenario"]["clusters"]), level=["onw", "clusters"], axis=1)
    
    twh = df.drop(["Generator", "Link", "Line", "StorageUnit"]).div(1e6)  # TWh

    gw = df.drop(["Store", "Line"]).div(1e3)  # GW

    twh = twh.groupby(level=1).sum()

    gen = gw.loc[["Generator", "StorageUnit"]].groupby(level=1).sum()

    con = gw.loc[["Link"]].groupby(level=1).sum()

    return gen, con, twh


# Process energy balances
def energy_balances(scenarios, sector, country, onw=100):
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/postnetworks/base_s_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in config["scenario"]["ll"]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }

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
    
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances = {i.replace(" ", "_"): [i] for i in energy_balance.index.levels[0]}
    balances["energy"] = [
        i for i in energy_balance.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():
        if k == sector:
            df = energy_balance.loc[energy_balance.index.get_level_values(0).isin(v)]
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

            df.columns = pd.MultiIndex.from_tuples(
                [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "year"]
            )

            df = df.xs((onw, config["scenario"]["clusters"]), level=["onw", "clusters"], axis=1)

    return df 


# PProcess total costs
def costs(scenarios, country, onw=100):
    files = sorted([f for f in os.listdir(f"{scenarios}/configs") if os.path.isfile(os.path.join(f"{scenarios}/configs", f))])
    if files:
        with open(os.path.join(f"{scenarios}/configs", files[0]), 'r') as file:
            config = yaml.safe_load(file)

    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): f"{scenarios}"
        + f"/postnetworks/base_s_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for cluster in config["scenario"]["clusters"]
        for opt in config["scenario"]["opts"]
        for sector_opt in config["scenario"]["sector_opts"]
        for ll in config["scenario"]["ll"]
        for planning_horizon in config["scenario"]["planning_horizons"]
    }
    
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )

    cost = pd.DataFrame(columns=columns, dtype=float)

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

        cost = cost.reindex(df_capital_annual.index.union(cost.index))
        cost.loc[df_capital_annual.index, label] = df_capital_annual

        if country == "EU":
            df_marginal_costs_annual = pd.concat([n.statistics.opex()], keys=["marginal"])
        else:
            df_marginal_costs_annual = pd.concat([n.statistics.opex(groupby=["carrier", "bus_carrier", "country"])], keys=["marginal"])
            idx = pd.IndexSlice
            df_marginal_costs_annual = df_marginal_costs_annual.loc[idx[:,:,:,:,country]]
            df_marginal_costs_annual = df_marginal_costs_annual.reorder_levels([None, 'bus_carrier', 'component', 'carrier'])
            df_marginal_costs_annual = df_marginal_costs_annual.sort_index()

        cost = cost.reindex(df_marginal_costs_annual.index.union(cost.index))
        cost.loc[df_marginal_costs_annual.index, label] = df_marginal_costs_annual

    cost = cost.fillna(0)

    if country == "EU":
        df = cost.groupby(cost.index.get_level_values(2)).sum()
    else:
        df = cost.groupby(cost.index.get_level_values(3)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    df.columns = pd.MultiIndex.from_tuples(
                [parse_index(c) for c in df.columns], names=["clusters", "lv", "onw", "year"]
            )

    df = df.xs((onw, config["scenario"]["clusters"]), level=["onw", "clusters"], axis=1)
    
    return df


# Settings
# Set path for scenarios
MAIN_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-default-2025-2050-5-T-H-B-I-A"
LOWCARBON_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-low_carbon_budget-2025-2050-5-T-H-B-I-A"
NOH2GRID_SCENARIOS = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/myopic-noh2grid-2025-2050-5-T-H-B-I-A"

# Config settings for scenario
SCENARIOS = {
    (0, 0): (MAIN_SCENARIOS, 100), 
    (1, 0): (LOWCARBON_SCENARIOS, 100),
    (0, 1): (NOH2GRID_SCENARIOS, 100),
}
NAMES = ["low_carbon", "no_h2grid"]
countries = ["DE", "EU"] # EU means all European countries

for country in countries:
    # Cost preparation
    cost = pd.concat(
        {k: costs(scenarios, country, onw) for k, (scenarios, onw) in SCENARIOS.items()},
        names=NAMES,
    )
    cost.index.names = cost.index.names[:-1] + ["carrier"]
    cost = cost.stack([0, 1]).to_xarray()
    cost.name = "costs"

    # Energy balance preparation
    energy = pd.concat(
        {k: energy_balances(scenarios, "energy", country, onw) for k, (scenarios, onw) in SCENARIOS.items()},
        names=NAMES,
    )

    energy.index.names = energy.index.names[:-1] + ["carrier"]
    energy = energy.stack([0, 1]).to_xarray()
    energy.name = "energy"

    # CO2 balance preparation
    if country == "EU":
        co2 = pd.concat(
            {k: energy_balances(scenarios, "co2", country, onw) for k, (scenarios, onw) in SCENARIOS.items()},
            names=NAMES,
        )

        co2.index.names = co2.index.names[:-1] + ["carrier"]
        co2 = co2.stack([0, 1]).to_xarray()
        co2.name = "co2"

    # H2 balance preparation
    h2 = pd.concat(
        {k: energy_balances(scenarios, "Hydrogen_Storage", country, onw) for k, (scenarios, onw) in SCENARIOS.items()},
        names=NAMES,
    )

    h2.index.names = h2.index.names[:-1] + ["carrier"]
    h2 = h2.stack([0, 1]).to_xarray()
    h2.name = "hydrogen"

    # Generation, storage and conversion capacity preparation
    cap = pd.concat(
        {
            k: pd.concat(
                capacity(scenarios, country, onw),
                keys=["generation", "conversion", "storage"],
            )
            for k, (scenarios, onw) in SCENARIOS.items()
        },
        names=NAMES,
    )

    cap.index.names = cap.index.names[:-2] + ["category", "carrier"]
    cap = cap.stack([0, 1]).unstack("category").to_xarray()
    
    # Merge and output all data
    if country == "EU":
        ds = xr.merge([cost, energy, co2, h2, cap]).round(2)
    else:
        ds = xr.merge([cost, energy, h2, cap]).round(2)
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{country}_scenarios_streamlit.nc", encoding=encoding)