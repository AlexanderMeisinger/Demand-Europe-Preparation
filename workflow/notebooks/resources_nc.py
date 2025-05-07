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
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from pypsa.descriptors import get_switchable_as_dense as as_dense

from shapely import wkt
import sys, os


PATH = "../"
SCRIPTS_PATH = "pypsa-eur/scripts/"

sys.path.append(os.path.join(PATH, SCRIPTS_PATH))
from _helpers import rename_techs

xr.set_options(display_style="html")

OUTPUT = "workflow/results/scenario"

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
    #"H2": "RdPu",
    "heat": "Reds",
    "solid biomass": "Greens",
    #"gas": "Oranges",
    "H2": "Oranges",
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



def plot_regional_demands(df, geodf, carrier, add, year, series=None, vmax=None, vmin=0):
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1)#.drop("EU")

    proj = ccrs.EqualEarth()
    geodf = geodf.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    geodf.plot(
        ax=ax,
        column=series,
        # transform=ccrs.PlateCarree(),
        cmap=cmap[carrier],
        #cmap=custom_cmap,
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

    plt.savefig(f"{OUTPUT}/demand-map-{carrier}-{add}-{year}.pdf", bbox_inches="tight")


def plot_regional_import_export(df, geodf, carrier, add, year, series=None, vmax=None, vmin=None):
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1)#.drop("EU")

    proj = ccrs.EqualEarth()
    geodf = geodf.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    # Create two segments of the colormap
    blues = cm.get_cmap("Blues", 128)
    oranges = cm.get_cmap("Oranges_r", 128)  # reverse for light near 0

    # Combine: oranges (negative) → white → blues (positive)
    combined = np.vstack((
        oranges(np.linspace(0, 1, 128)),
        blues(np.linspace(0, 1, 128))
    ))
    custom_cmap = ListedColormap(combined)

    geodf.plot(
        ax=ax,
        column=series,
        # transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        #cmap=cmap[carrier],
        linewidths=0,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": "Hydrogen Trade [TWh/a]", "shrink": 0.7, "extend": "max"},
    )

    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"), linewidth=0.2, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS.with_scale("50m"), linewidth=0.2, zorder=2)

    ax.set_frame_on(False)
    ax.set_facecolor("white")

    plt.savefig(f"{OUTPUT}/h2-trade-{carrier}-{add}-{year}.pdf", bbox_inches="tight")

# Process energy balances
def energy_balances(filename, sector):
    # Read nc file
    n = pypsa.Network(filename)
    # Extract and sort results from nc file
    df_annual = n.statistics.energy_balance(groupby=["carrier", "bus_carrier", "bus"])

    df_annual = df_annual.reorder_levels(['bus_carrier', 'component', 'carrier', "bus"])
    df_annual = df_annual.sort_index()

    energy_balance = df_annual
    
    # Fill nan values
    energy_balance = energy_balance.fillna(0)
    
    # Get sectors
    balances = {i.replace(" ", "_"): [i] for i in energy_balance.index.levels[0]}
    # Define sector energy
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    balances["energy"] = [
        i for i in energy_balance.index.levels[0] if i not in co2_carriers
    ]

    for k, v in balances.items():
        if k == sector:
            # Filter results
            df = energy_balance.loc[energy_balance.index.get_level_values(0).isin(v)]
            # Convert MWh to TWh
            df = df / 1e6
    return df 


# Settings
scenarios = {
    "main": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.main/networks/base_s_39__144H_{year}.nc",
    "lowcarbon": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.lowcarbon/networks/base_s_39__144H-cb12.0ex0_{year}.nc",
    "lowH2cost": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.lowH2cost/networks/base_s_39__144H_{year}.nc",
    "gridfreeze": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.gridfreeze/networks/base_s_39__144H_{year}.nc",
    "highH2demand": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.highH2demand/networks/base_s_39__144H_{year}.nc",
    "highcarbon": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.highcarbon/networks/base_s_39__144H_{year}.nc",
}

years = [2025, 2030, 2035, 2040, 2045, 2050]

fn = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/resources/myopic/config.main/regions_onshore_base_s_39.geojson"
nodes = gpd.read_file(fn).set_index("name")

for name, path_template in scenarios.items():
    print(name)
    for year in years:
        path = path_template.format(year=year)

        # H2 balance preparation
        h2 = energy_balances(path, "Hydrogen_Storage")
        h2 = h2.reset_index()
        h2 = h2.rename(columns={0: "H2"})
        h2["bus"] = h2["bus"].str.replace(r"\s+H2$", "", regex=True)
        import_export = h2[["bus", "H2", "carrier"]]
        h2 = h2[["bus", "H2"]] 

        demand = h2[h2["H2"]<0]
        demand.loc[:,"H2"] = demand["H2"]*(-1)
        #supply = h2[h2["H2"]>0]
        import_export = import_export[import_export["carrier"].isin(["H2 pipeline", "H2 pipeline retrofitted"])]
        import_export = import_export[["bus","H2"]]
        import_export.loc[:,"H2"] = import_export["H2"]*(-1)

        demand_country = demand.groupby("bus", as_index=False).sum()
        demand_country =demand_country.set_index("bus")
        #supply_country = supply.groupby("bus", as_index=False).sum()
        #supply_country = supply_country.set_index("bus")
        import_export_country = import_export.groupby("bus", as_index=False).sum()
        import_export_country = import_export_country.set_index("bus")

        plot_regional_demands(demand_country, nodes, "H2", name, year)
        if name != "gridfreeze":
            plot_regional_import_export(import_export_country, nodes, "H2", name, year)