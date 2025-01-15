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
from plot_summary import rename_techs

plt.style.use(["bmh", "matplotlibrc"])
xr.set_options(display_style="html")

CLUSTERS = 181

MAIN_SCENARIOS = PATH + "results/20221227-main"
IMP_SCENARIOS = PATH + "results/20221227-import"
SHP_SCENARIOS = PATH + "results/20221227-shipping"
COST_SCENARIOS = PATH + "results/20221227-costs"
	

def parse_index(c, with_resolution=False):
    clusters = c[0]

    lv = c[1]

    match = re.search(r"onwind\+p([0-9.]*)", c[2])
    onw = 100.0 if match is None else 100 * float(match.groups()[0])

    h2 = "no H2 grid" if "noH2network" in c[2] else "H2 grid"

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

    if clusters is None:
        clusters = CLUSTERS

    horizon = "2030" if "rev0" in scenarios else "2050"

    costs = pd.read_csv(
        scenarios + f"/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2]
    )

    costs = costs.xs(horizon, level="planning_horizon", axis=1)

    names = ["clusters", "lv", "onw", "h2"]
    if with_resolution:
        names += ("res",)

    costs.columns = pd.MultiIndex.from_tuples(
        [parse_index(c, with_resolution) for c in costs.columns], names=names
    )

    if not with_space:
        costs = costs.xs(str(clusters), level="clusters", axis=1)

    df = costs.groupby(level=2).sum().div(1e9)

    to_drop = df.index[df.max(axis=1).fillna(0.0) < 1.2]
    print(to_drop)
    df.drop(to_drop, inplace=True)

    if "-imp" in scenarios:
        # imports for methanol, kerosene and naphtha at 120 €/MWh
        print("add import costs")
        df.loc["green e-fuel imports"] = (1026.64 + 546.36) * 120e6 / 1e9  # bn€/a

    return df

    
SCENARIOS = {
    (0, 0, 0, 0): (MAIN_SCENARIOS, 100),
    (1, 0, 0, 0): (COST_SCENARIOS, 100),
    (0, 1, 0, 0): (IMP_SCENARIOS, 100),
    (0, 0, 1, 0): (SHP_SCENARIOS, 100),
    (0, 0, 0, 1): (MAIN_SCENARIOS, 0),
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

ds = xr.merge([tsc, energy, co2, cap]).round(2)
comp = dict(zlib=True, complevel=9)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("scenarios.nc", encoding=encoding)

