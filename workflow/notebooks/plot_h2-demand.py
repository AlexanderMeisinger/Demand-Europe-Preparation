# This script plots the h2 demand from PyPSA-Eur
# Author: Alexander Meisinger
# Project: H2Global meets Africa (FENES, OTH Regensburg)
# Base: https://doi.org/10.1016/j.joule.2023.06.016

# Package imports
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


def plot_regional_demands(df, geodf, carrier, add, year, series=None, vmax=None, vmin=0):
    """
    Plot a map of regional energy demand for a specified energy carrier.

    Parameters:
        df (pd.DataFrame): DataFrame containing energy data by region.
        geodf (gpd.GeoDataFrame): GeoDataFrame containing region geometries.
        carrier (str): Type of energy carrier to visualize (e.g., "H2").
        add (str): Scenario name used in the filename.
        year (int): Year of analysis.
        series (pd.Series, optional): Precomputed series of values to plot.
        vmax (float, optional): Maximum value for color scale.
        vmin (float, optional): Minimum value for color scale.
    """
    # If no custom series is provided, aggregate the matching columns using the defined regex pattern
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1)

    # Set map projection to Equal Earth (for visually balanced global maps)
    proj = ccrs.EqualEarth()

    # Reproject the geodataframe to match the selected map projection
    geodf = geodf.to_crs(proj.proj4_init)

    # Create figure and axes for plotting with a map projection
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    # Plot regional values on the map using color shading based on demand values
    geodf.plot(
        ax=ax,
        column=series,
        cmap=cmap[carrier],
        linewidths=0,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": title[carrier], "shrink": 0.7, "extend": "max"},
    )

    # Add coastlines and country borders for geographic context
    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"), linewidth=0.2, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS.with_scale("50m"), linewidth=0.2, zorder=2)

    # Remove plot frame and set background color
    ax.set_frame_on(False)
    ax.set_facecolor("white")

    # Save the plot as a PDF to the configured OUTPUT directory
    plt.savefig(f"{OUTPUT}/demand-map-{carrier}-{add}-{year}.pdf", bbox_inches="tight")


def plot_regional_import_export(df, geodf, carrier, add, year, series=None, vmax=None, vmin=None):
    """
    Plot a map showing regional hydrogen trade (import/export) using a diverging color scale.

    Parameters:
        df (pd.DataFrame): DataFrame of import/export values by region.
        geodf (gpd.GeoDataFrame): GeoDataFrame with region geometries.
        carrier (str): Energy carrier type (e.g., "H2").
        add (str): Scenario name for file output.
        year (int): Year of data.
        series (pd.Series, optional): Series of trade values. If None, it is computed.
        vmax (float, optional): Upper bound for color normalization.
        vmin (float, optional): Lower bound for color normalization.
    """
    # If no series provided, compute total trade per region using regex match
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1)

    # Set map projection to Equal Earth for balanced visual presentation
    proj = ccrs.EqualEarth()

    # Reproject GeoDataFrame to match the chosen map projection
    geodf = geodf.to_crs(proj.proj4_init)

    # Initialize plot with map projection
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    # Create two separate color segments: one for imports (negative), one for exports (positive)
    blues = cm.get_cmap("Blues", 128)
    oranges = cm.get_cmap("Oranges_r", 128)

    # Stack the two segments to form a diverging colormap (orange → white → blue)
    combined = np.vstack((
        oranges(np.linspace(0, 1, 128)),
        blues(np.linspace(0, 1, 128))
    ))
    custom_cmap = ListedColormap(combined)

    # Plot the data as colored regions based on trade values
    geodf.plot(
        ax=ax,
        column=series,
        cmap=custom_cmap,
        linewidths=0,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": "Hydrogen Trade [TWh/a]", "shrink": 0.7, "extend": "max"},
    )

    # Add map features: coastlines and borders
    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"), linewidth=0.2, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS.with_scale("50m"), linewidth=0.2, zorder=2)

    # Remove the map frame and set background to white
    ax.set_frame_on(False)
    ax.set_facecolor("white")

    # Save plot to PDF using scenario, carrier, and year for filename
    plt.savefig(f"{OUTPUT}/h2-trade-{carrier}-{add}-{year}.pdf", bbox_inches="tight")


def energy_balances(filename, sector):
    """
    Extract and process energy balance data for a specific sector from a PyPSA network file.

    Parameters:
        filename (str): Path to the NetCDF file containing the PyPSA network.
        sector (str): Energy sector of interest (e.g., "Hydrogen_Storage").

    Returns:
        pd.DataFrame: Processed DataFrame of annual energy balances for the sector.
    """
    # Load the PyPSA network from a .nc file
    n = pypsa.Network(filename)

    # Extract energy balances grouped by carrier and bus info
    df_annual = n.statistics.energy_balance(groupby=["carrier", "bus_carrier", "bus"])

    # Reorder and sort index levels for consistent access
    df_annual = df_annual.reorder_levels(['bus_carrier', 'component', 'carrier', "bus"])
    df_annual = df_annual.sort_index()
    
    # Fill missing values (if any) with 0
    energy_balance = df_annual
    energy_balance = energy_balance.fillna(0)
    
    # Map index level names (bus_carrier) to cleaned sector keys
    balances = {i.replace(" ", "_"): [i] for i in energy_balance.index.levels[0]}
    
    # Define general energy carriers (exclude CO₂-related ones)
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    balances["energy"] = [
        i for i in energy_balance.index.levels[0] if i not in co2_carriers
    ]

    # Select and Return Sector Data
    for k, v in balances.items():
        if k == sector:
            # Filter results
            df = energy_balance.loc[energy_balance.index.get_level_values(0).isin(v)]
            # Convert MWh to TWh
            df = df / 1e6
    return df 


def plot_regional_H2_demand_FT(df, geodf, carrier, add, year, series=None, vmax=None, vmin=None):
    """
    Plot a map showing regional hydrogen demand for Fischer-Tropsch processes.

    Parameters:
        df (pd.DataFrame): DataFrame of H2 FT demand values by region.
        geodf (gpd.GeoDataFrame): GeoDataFrame with region geometries.
        carrier (str): Type of energy carrier ("H2").
        add (str): Scenario name for use in the filename.
        year (int): Year of the data.
        series (pd.Series, optional): Precomputed series. If None, it's aggregated from df.
        vmax (float, optional): Maximum value for color scale.
        vmin (float, optional): Minimum value for color scale.
    """
    # If no specific data series is given, extract demand columns using regex and aggregate
    if series is None:
        series = df.filter(regex=regex[carrier]).sum(axis=1)

    # Use Equal Earth projection for global spatial balance
    proj = ccrs.EqualEarth()

    # Reproject the geodataframe to match the map projection
    geodf = geodf.to_crs(proj.proj4_init)

    # Create figure and axes with geographic projection
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": proj})

    # Two color maps: blues for positive (high demand), oranges reversed for low demand
    blues = cm.get_cmap("Blues", 128)
    oranges = cm.get_cmap("Oranges_r", 128)  # reverse for light near 0

    # Combine: oranges (negative) → white → blues (positive)
    combined = np.vstack((
        oranges(np.linspace(0, 1, 128)),
        blues(np.linspace(0, 1, 128))
    ))
    custom_cmap = ListedColormap(combined)

    # Plotting the Demand Map
    geodf.plot(
        ax=ax,
        column=series,
        cmap=custom_cmap,
        linewidths=0,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": "H2-FT Demand [TWh/a]", "shrink": 0.7, "extend": "max"},
    )

    # Add contextual map features
    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"), linewidth=0.2, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS.with_scale("50m"), linewidth=0.2, zorder=2)

    # Clean up map frame and background
    ax.set_frame_on(False)
    ax.set_facecolor("white")

    # Save the plot to the results directory using scenario name and yea
    plt.savefig(f"{OUTPUT}/H2-FT-demand-{carrier}-{add}-{year}.pdf", bbox_inches="tight")


# Scenario Configuration and Input Data
# Define scenarios and corresponding file path templates for each year
scenarios = {
    "main": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.main/networks/base_s_39__144H_{year}.nc",
    "lowcarbon": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.lowcarbon/networks/base_s_39__144H-cb12.0ex0_{year}.nc",
    "lowH2cost": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.lowH2cost/networks/base_s_39__144H_{year}.nc",
    "gridfreeze": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.gridfreeze/networks/base_s_39__144H_{year}.nc",
    "highH2demand": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.highH2demand/networks/base_s_39__144H_{year}.nc",
    "highcarbon": "/mnt/e/H2GMA/Github/Europe/pypsa-eur/results/myopic/config.highcarbon/networks/base_s_39__144H_{year}.nc",
}

# List of planning horizon years to iterate over
years = [2025, 2030, 2035, 2040, 2045, 2050]

# Load geographic nodes from GeoJSON and set index to region name
fn = "/mnt/e/H2GMA/Github/Europe/pypsa-eur/resources/myopic/config.main/regions_onshore_base_s_39.geojson"
nodes = gpd.read_file(fn).set_index("name")

# Directory for saving scenario result visualizations
OUTPUT = "workflow/results/scenario"

# Set xarray to display outputs in HTML (nicer formatting in Jupyter)
xr.set_options(display_style="html")

# Plot titles by carrier
title = {
    "electricity": "Electricity Demand [TWh/a]",
    "H2": "Hydrogen Demand [TWh/a]",
    "heat": "Heat Demand [TWh/a]",
    "solid biomass": "Solid Biomass Demand [TWh/a]",
    "gas": "Methane Demand [TWh/a]",
    "oil": "Liquid Hydrocarbons Demand [TWh/a]",
    "process emission": "Process Emissions [MtCO2/a]",
}

# Colormaps used for plotting each carrier
cmap = {"H2": "Oranges",}

# Patterns used to extract specific carriers
regex = {"H2": r"(H2|fuel cell)",}


# Main Loop: Process and Visualize Data
for name, path_template in scenarios.items():
    print(name)
    for year in years:
        path = path_template.format(year=year)

        # Load H2 Energy Data
        h2 = energy_balances(path, "Hydrogen_Storage")
        h2 = h2.reset_index()
        h2 = h2.rename(columns={0: "H2"})
        h2["bus"] = h2["bus"].str.replace(r"\s+H2$", "", regex=True)

        # Prepare datasets for different uses
        import_export = h2[["bus", "H2", "carrier"]]
        H2_FT_demand = h2[["bus", "H2", "carrier"]]
        h2 = h2[["bus", "H2"]] 

        # Demand Processing
        demand = h2[h2["H2"]<0]
        demand.loc[:,"H2"] = demand["H2"]*(-1)
        #supply = h2[h2["H2"]>0]

        # Import/Export Processing
        import_export = import_export[import_export["carrier"].isin(["H2 pipeline", "H2 pipeline retrofitted"])]
        import_export = import_export[["bus","H2"]]
        import_export.loc[:,"H2"] = import_export["H2"]*(-1)

        # Aggregate to Country Level
        demand_country = demand.groupby("bus", as_index=False).sum()
        demand_country =demand_country.set_index("bus")
        #supply_country = supply.groupby("bus", as_index=False).sum()
        #supply_country = supply_country.set_index("bus")
        import_export_country = import_export.groupby("bus", as_index=False).sum()
        import_export_country = import_export_country.set_index("bus")

        # Plot Demand and Trade Maps
        plot_regional_demands(demand_country, nodes, "H2", name, year)
        if name != "gridfreeze":
            plot_regional_import_export(import_export_country, nodes, "H2", name, year)

        # Preparation Hydrogen Demand for Fischer-Tropsch
        H2_FT_demand = H2_FT_demand[H2_FT_demand["carrier"].isin(["Fischer-Tropsch"])]
        H2_FT_demand = H2_FT_demand[["bus","H2"]]
        H2_FT_demand.loc[:,"H2"] = H2_FT_demand["H2"]*(-1)
        H2_FT_demand = H2_FT_demand.groupby("bus", as_index=False).sum()
        H2_FT_demand = H2_FT_demand.set_index("bus")

        # Plot FT-specific hydrogen demand
        plot_regional_H2_demand_FT(H2_FT_demand, nodes, "H2", name, year)