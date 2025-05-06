import pypsa
import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs
import cartopy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Patch
from matplotlib.legend_handler import HandlerPatch

from pypsa.descriptors import get_switchable_as_dense as as_dense

from shapely import wkt
import sys, os


PATH = "../"
SCRIPTS_PATH = "pypsa-eur/scripts/"

sys.path.append(os.path.join(PATH, SCRIPTS_PATH))
from _helpers import rename_techs
#from plot_network import assign_location
#from _helpers import override_component_attrs
#from build_gas_input_locations import build_gas_input_locations, load_bus_regions

xr.set_options(display_style="html")


OUTPUT = "workflow/results/test"
RUN = "myopic-default-20250202-2025-2050-5-T-H-B-I-A"

CLUSTERS = 39
OPTS = '144H-T-H-B-I-A'
planning_year = 2050
SCENARIO = f"base_s_{CLUSTERS}__{OPTS}_{planning_year}"
OVERRIDES = PATH + "pypsa-eur/data/override_component_attrs"

with open("config/test/config.myopic_play.yaml") as file:
    config = yaml.safe_load(file)


fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_onshore_base_s_{CLUSTERS}.geojson"
nodes = gpd.read_file(fn).set_index("name")

fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_offshore_base_s_{CLUSTERS}.geojson"
offnodes = gpd.read_file(fn).set_index("name")

fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/country_shapes.geojson"
cts = gpd.read_file(fn).set_index("name")

regions = pd.concat(
    [
        gpd.read_file(f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_onshore.geojson"),
        gpd.read_file(f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_offshore.geojson"),
    ]
)
regions = regions.dissolve("name")

fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_onshore.geojson"
onregions = gpd.read_file(fn).set_index("name")

fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/regions_onshore.geojson"
offregions = gpd.read_file(fn).set_index("name")

epsg = 3035
regions["Area"] = regions.to_crs(epsg=epsg).area.div(1e6)
onregions["Area"] = onregions.to_crs(epsg=epsg).area.div(1e6)
offregions["Area"] = offregions.to_crs(epsg=epsg).area.div(1e6)
nodes["Area"] = nodes.to_crs(epsg=epsg).area.div(1e6)


europe_shape = nodes.dissolve()
europe_shape.index = ["EU"]


minx, miny, maxx, maxy = europe_shape.explode(ignore_index=True).total_bounds
BOUNDARIES = [minx, maxx - 4, miny, maxy]


fn = f"{PATH}/pypsa-eur/results/myopic/{RUN}/networks/{SCENARIO}.nc"
n = pypsa.Network(fn)


unique_link_carriers = n.links.carrier.unique()
GAS_NETWORK = "gas pipeline" in unique_link_carriers
H2_NETWORK = any("H2 pipeline" in ulc for ulc in unique_link_carriers)


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    # if "heat pump" in tech or "resistive heater" in tech:
    #    return "power-to-heat"
    # elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
    #    return "power-to-gas"
    if tech == "H2":
        return "H2 storage"
    # elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
    #    return "gas-to-power/heat"
    # elif "solar" in tech:
    #    return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    #    if "heat pump" in tech:
    #        return "heat pump"
    elif tech == "gas":
        return "fossil gas"
    # elif "CC" in tech or "sequestration" in tech:
    #    return "CCS"
    elif tech in ["industry electricity", "agriculture electricity"]:
        return "industry electricity"
    elif "oil emissions" in tech:
        return "oil emissions"
    else:
        return tech

title = {
    "electricity": "Electricity Demand [TWh/a]",
    "H2": "Hydrogen Demand [TWh/a]",
    "heat": "Heat Demand [TWh/a]",
    "solid biomass": "Solid Biomass Demand [TWh/a]",
    "gas": "Methane Demand [TWh/a]",
    "oil": "Liquid Hydrocarbons Demand [TWh/a]",
    "process emission": "Process Emissions [MtCO2/a]",
}

cmap = {
    "electricity": "Blues",
    "H2": "RdPu",
    "heat": "Reds",
    "solid biomass": "Greens",
    "gas": "Oranges",
    "oil": "Greys",
    "process emission": "Greys",
}

regex = {
    "electricity": r"(electricity|EV)",
    "H2": r"(H2|fuel cell)",
    "heat": r"heat",
    "solid biomass": r"biomass",
    "oil": r"( oil|naphtha|kerosene|methanol)",
    "gas": r"gas",
}


def plot_regional_demands(df, geodf, carrier, series=None, vmax=None, vmin=0):
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1).drop("EU")

    proj = ccrs.EqualEarth()
    geodf = geodf.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    geodf.plot(
        ax=ax,
        column=series,
        # transform=ccrs.PlateCarree(),
        cmap=cmap[carrier],
        linewidths=0,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": title[carrier], "shrink": 0.7, "extend": "max"},
    )

    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"), linewidth=0.2, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS.with_scale("50m"), linewidth=0.2, zorder=2)

    ax.set_frame_on(False)
    ax.set_facecolor("white")

    plt.savefig(f"{OUTPUT}/demand-map-{carrier}.pdf", bbox_inches="tight")

# Double check
def get_biomass_demand():
    fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/industrial_energy_demand_base_s_{CLUSTERS}_{planning_year}.csv"
    industrial_demand = pd.read_csv(fn, index_col=0)

    return industrial_demand["solid biomass"]


def get_process_emission():
    fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/industrial_energy_demand_base_s_{CLUSTERS}_{planning_year}.csv"
    industrial_demand = pd.read_csv(fn, index_col=0)

    return industrial_demand["process emission"]

# ToDo: Double check oil demand - maybe do it with n.statistics
# ToDo: For now, only loads - later maybe add links
# Check energy_supply and EU demands
def get_oil_demand_energy_statistics():
    test = n.statistics.energy_balance()
    # EU: agriculture machinery oil
    # EU: gas for industry
    # EU: industry methanol
    # EU: kerosene for aviation
    # EU: naphtha for industry
    # EU: process emissions
    # EU: shipping methanol
    # EU: solid biomass for industry
    # EU: co2 sequestered
    # EU: co2
    # EU: gas
    # Or only focus on load? or also on link?


def get_oil_demand(with_methanol=False):
    fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/industrial_energy_demand_base_s_{CLUSTERS}_{planning_year}.csv"
    industrial_demand = pd.read_csv(fn, index_col=0)

    fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/pop_weighted_energy_totals_s_{CLUSTERS}.csv"
    nodal_energy_totals = pd.read_csv(fn, index_col=0)

    oil = [
        "total international aviation",
        "total domestic aviation",
        "total agriculture machinery",
    ]
    if with_methanol:
        oil.append("total domestic navigation")
        #ToDo: total international navigation?
        #ToDo: rail?

    demand = industrial_demand["naphtha"] + nodal_energy_totals[oil].sum(axis=1)

    if with_methanol:
        fn = f"{PATH}/pypsa-eur/resources/myopic/{RUN}/shipping_demand_s_{CLUSTERS}.csv"
        efficiency = (
            config["sector"]["shipping_oil_efficiency"]
            / config["sector"]["shipping_methanol_efficiency"]
        )
        demand += pd.read_csv(fn, index_col=0).squeeze() * efficiency

    return demand


demand = as_dense(n, "Load", "p_set").div(1e6)  # TWh
demand_grouped = demand.groupby(
    [n.loads.carrier, n.loads.bus.map(n.buses.location)], axis=1
    ).sum()
demand_by_region = (n.snapshot_weightings.generators @ demand_grouped).unstack(level=0)

get_oil_demand_energy_statistics()
plot_regional_demands(demand_by_region, nodes, "electricity", vmax=140)
plot_regional_demands(demand_by_region, nodes, "H2", vmax=40)
plot_regional_demands(demand_by_region, nodes, "heat", vmax=100)
plot_regional_demands(demand_by_region, nodes, "solid biomass", series=get_biomass_demand(), vmax=25)
plot_regional_demands(demand_by_region, nodes, "oil", series=get_oil_demand(with_methanol=True), vmax=100)
plot_regional_demands(demand_by_region, nodes, "gas", vmax=16)

# Missing: ammonia and methanol

# Check out industry demand

mapping = {
    "H2 for industry": "hydrogen",
    "H2 for shipping": "hydrogen",
    "agriculture electricity": "electricity",
    "agriculture heat": "heat",
    "agriculture machinery oil": "oil",
    "agriculture machinery oil emissions": "emissions",
    "electricity": "electricity",
    "gas for industry": "methane",
    "industry electricity": "electricity",
    "kerosene for aviation": "oil",
    "land transport EV": "electricity",
    "land transport fuel cell": "hydrogen",
    "low-temperature heat for industry": "heat",
    "naphtha for industry": "oil",
    "industry methanol": "methanol", #H2G-A: Double check
    "shipping methanol": "methanol",
    "shipping methanol emissions": "emissions",
    "oil emissions": "emissions",
    "process emissions": "emissions",
    "residential rural heat": "heat",
    "residential urban decentral heat": "heat",
    "services rural heat": "heat",
    "services urban decentral heat": "heat",
    "solid biomass for industry": "solid biomass",
    "urban central heat": "heat", 
    "rural heat": "heat", #H2G-A: Double check
    "urban decentral heat": "heat", #H2G-A: Double check
}

mapping_sector = {
    "H2 for industry": ("hydrogen", "industry"),
    "H2 for shipping": ("hydrogen", "shipping"),
    "agriculture electricity": ("electricity", "agriculture"),
    "agriculture heat": ("heat", "agriculture"),
    "agriculture machinery oil": ("oil", "agriculture"),
    "agriculture machinery oil emissions": ("emissions", "agriculture"),
    "electricity": ("electricity", "residential"),
    "gas for industry": ("methane", "industry"),
    "industry electricity": ("electricity", "industry"),
    "kerosene for aviation": ("oil", "aviation"),
    "land transport EV": ("electricity", "land transport"),
    "land transport fuel cell": ("hydrogen", "land transport"),
    "low-temperature heat for industry": ("heat", "industry"),
    "naphtha for industry": ("oil", "industry"),
    "industry methanol": ("methanol", "industry"),
    "shipping methanol": ("methanol", "shipping"),
    "shipping methanol emissions": ("emissions", "other"),
    "oil emissions": ("emissions", "other"),
    "process emissions": ("emissions", "process"),
    "residential rural heat": ("heat", "residential rural"),
    "residential urban decentral heat": ("heat", "residential urban"),
    "services rural heat": ("heat", "services rural"),
    "services urban decentral heat": ("heat", "services urban"),
    "solid biomass for industry": ("solid biomass", "industry"),
    "urban central heat": ("heat", "district heating"),
    "rural heat": ("heat-rural", "district heating"), #H2G-A: Double check
    "urban decentral heat": ("heat-decentral", "district heating"), #H2G-A: Double check
}

order = [
    "electricity",
    "heat",
    "oil",
    "methanol",
    "solid biomass",
    "methane",
    "hydrogen",
]

df = demand_by_region.sum()

mapping

df.index = pd.MultiIndex.from_tuples([(mapping[i], i) for i in df.index])

df.drop("emissions", inplace=True)

colors = config["plotting"]["tech_colors"]
colors["solid biomass"] = "seagreen"
colors["methane"] = "#db6a25"

pd.DataFrame(df.groupby(level=0).sum().div(1e3).loc[order], columns=[""]).T.plot.bar(
    stacked=True,
    color=colors,
    figsize=(2, 5),
    ylim=[0, 20],
    ylabel="Final energy an non-energy demand [1000 TWh]",
)
plt.savefig(f"{OUTPUT}/total-annual-demand.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 5))

df.unstack().loc[order].plot.bar(
    ax=ax, stacked=True, cmap="tab20", edgecolor="k", ylim=(-100, 4500)
)

plt.legend(bbox_to_anchor=(1, 1))

plt.ylabel("Final energy and non-energy demand [TWh/a]")

plt.savefig(f"{OUTPUT}/demand-by-carrier.pdf")

df = demand_by_region.sum()

df.index = pd.MultiIndex.from_tuples([mapping_sector[i] for i in df.index])

df = df.loc[df > 0]

df

fig, ax = plt.subplots(figsize=(4.5, 4.5))

df.unstack().loc[order].T.plot.barh(
    ax=ax, color=colors, stacked=True, edgecolor="k", xlim=(-100, 3500)
)

plt.legend(bbox_to_anchor=(1, 1))

plt.xlabel("Final energy and non-energy demand [TWh/a]")

plt.savefig(f"{OUTPUT}/demand-by-sector-carrier.pdf")

fig, ax = plt.subplots(figsize=(4.5, 4.5))

df.unstack().loc[order].plot.barh(
    ax=ax, stacked=True, cmap="tab20", edgecolor="k", xlim=(-100, 4500)
)

plt.legend(bbox_to_anchor=(1, 1))

plt.xlabel("Final energy and non-energy demand [TWh/a]")

plt.savefig(f"{OUTPUT}/demand-by-carrier-sector.pdf")

#if not GAS_NETWORK:
#    plot_system_demands(demand_by_region, europe_shape, "gas")


co2 = pd.read_csv(PATH + f"pypsa-eur/resources/myopic/{RUN}/co2_totals.csv", index_col=0) #H2G-A: Changed

fig, ax = plt.subplots(figsize=(7, 6))
co2.plot.barh(ax=ax, stacked=True, cmap="tab20")
plt.savefig(OUTPUT + "/co2.pdf")

fn = PATH + f"pypsa-eur/resources/myopic/{RUN}/industrial_production_per_country.csv"
iproduction_today = pd.read_csv(fn, index_col=0).sum()

fn = (
    PATH + f"pypsa-eur/resources/myopic/{RUN}/industrial_production_per_country_tomorrow_2050.csv"
)
iproduction_tomorrow = pd.read_csv(fn, index_col=0).sum()

fn = PATH + f"pypsa-eur/resources/myopic/{RUN}/industry_sector_ratios.csv"
iratios = pd.read_csv(fn, index_col=0)

fn = PATH + f"pypsa-eur/resources/myopic/{RUN}/industrial_energy_demand_per_country_today.csv"
ienergy_today = (
    pd.read_csv(
        fn,
        index_col=0,
        header=[0, 1],
    )
    .groupby(level=1, axis=1)
    .sum()
)

ienergy_today.rename({"gas": "methane"}, inplace=True)

iratios.rename({"elec": "electricity", "naphtha": "liquid"}, inplace=True)
iratios.loc["solid", :] = iratios.loc[["coke", "coal"]].sum()
iratios.drop(
    ["coke", "coal", "process emission", "process emission from feedstock"],
    axis=0,
    inplace=True,
)

ienergy_tomorrow = iratios * iproduction_tomorrow / 1e3
bc = [
    "Chlorine",
    "HVC",
    "HVC (mechanical recycling)",
    "HVC (chemical recycling)",
    "Methanol",
]

ienergy_tomorrow["Basic chemicals (without ammonia)"] = ienergy_tomorrow[bc].sum(axis=1)
ienergy_tomorrow.drop(bc, axis=1, inplace=True)
ienergy_today["DRI + Electric arc"] = 0.0
ienergy_today.loc["hydrogen", :] = 0.0
ienergy_tomorrow.loc["other", :] = 0.0
ienergy_tomorrow.loc["waste", :] = 0.0

ienergy_tomorrow.sort_index(axis=1, inplace=True)
ienergy_today.sort_index(axis=1, inplace=True)

ienergy_tomorrow.sort_index(axis=0, inplace=True)
ienergy_today.sort_index(axis=0, inplace=True)

tech_colors = config["plotting"]["tech_colors"]

tech_colors["electricity"] = "#ace37f"
tech_colors["hydrogen"] = "#f073da"

fig, ax = plt.subplots(figsize=(4, 6))

ienergy_tomorrow.T.plot.barh(
    ax=ax, stacked=True, width=0.3, color=tech_colors, position=1, edgecolor="k"
)
ienergy_today.T.plot.barh(
    ax=ax, stacked=True, width=0.3, color=tech_colors, position=0, edgecolor="k"
)


plt.xlabel("Final energy and non-energy [TWh/a]")
plt.xlim(-50, 1300)
plt.ylim(-1, 23)
handles, labels = ax.get_legend_handles_labels()
n = ienergy_today.shape[0]
plt.legend(handles[:n], labels[:n])
plt.savefig(OUTPUT + "/fec_industry_today_tomorrow.pdf", bbox_inches="tight")

df = ienergy_tomorrow.sum(axis=1).sort_values(ascending=False)
df = df.loc[df > 0]

fig, ax = plt.subplots(figsize=(3, 4))

df.plot.bar(ax=ax, width=0.3, edgecolor="k")

plt.ylabel("Final energy and non-energy [TWh/a]")
plt.xlabel("")

plt.savefig(OUTPUT + "/fec_industry_tomorrow_by_carrier.pdf", bbox_inches="tight")

pe = (
    pd.read_csv(
        PATH + f"pypsa-eur/resources/myopic/{RUN}/industry_sector_ratios.csv", index_col=0
    )
    .filter(like="process emission", axis=0)
    .sum()
)

pe_today = iproduction_today * pe / 1e3

pe_tomorrow = iproduction_tomorrow * pe / 1e3

fig, ax = plt.subplots(figsize=(4, 6))

pe_tomorrow.plot.barh(ax=ax, width=0.3, color="peachpuff", position=1, edgecolor="k")
pe_today.plot.barh(ax=ax, width=0.3, color="indianred", position=0, edgecolor="k")

plt.xlabel(f"Process emissions [MtCO$_2$/a]")
plt.xlim(-4, 110)
plt.ylim(-1, 27)
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    handles[:n],
    [
        f"2050:  {pe_tomorrow.sum():.0f} MtCO$_2$/a",
        f"today: {pe_today.sum():.0f} MtCO$_2$/a",
    ],
)
plt.savefig(OUTPUT + "/process-emissions.pdf", bbox_inches="tight")
