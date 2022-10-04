import sqlite3 as sql
from plot_params import *
from pandas import read_sql
import numpy as np

bins = 50

# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/event_location/test_plot/"

# data contains: charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width

# dataloading
with sql.connect(indir) as con:
    query = """
    SELECT
        dom_time, dom_x, dom_y, dom_z, event_no
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

hist, xedge, yedge = np.histogram2d(sql_data["dom_x"], sql_data["dom_y"], bins=bins)

plt.figure(figsize=single)
X, Y = np.meshgrid(xedge, yedge)
plt.pcolormesh(X, Y, hist, cmap = 'coolwarm')
plt.colorbar(label="triggers")
plt.savefig(outdir + "trigger_density_pmesh.png")

plt.figure(figsize=single)
# Creating plot
plt.hexbin(
    sql_data["dom_x"], sql_data["dom_y"], gridsize=bins, cmap = 'coolwarm'
    )
plt.colorbar(label="triggers")
plt.xlabel("x position")
plt.ylabel("x position")
plt.title(f"trigger density")
plt.savefig(outdir + "trigger_density.png")