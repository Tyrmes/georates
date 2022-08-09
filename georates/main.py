import pandas as pd
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from math import radians
import seaborn as sns
# %% Creation of synthetic field
# Generate random values
seed = 123
grid_resolution = 1
min_loc, max_loc = 0, 100
angle = radians(130)

x = y = np.arange(min_loc, max_loc, grid_resolution)
model = gs.Exponential(dim=2, var=2, len_scale=[20, 8], angles=angle)
# Since var is X and we will supress negative values, we need to move the mean so the
# actual var value is reflected in the variogram.
srf = gs.SRF(model, mean=10, seed=seed)
# Thickness
field = srf.structured([x, y])
# Properties such as net thickness, porosity, etc., do not have negative values
field = np.where(field < 0, 0, field)

#%% Creation of synthetic new wells
n_wells = 50
# Define the random number generator
rng = np.random.default_rng(seed)
# Define X, Y random locations
x_new = rng.integers(min_loc, max_loc, n_wells)
y_new = rng.integers(min_loc, max_loc, n_wells)
# Extract field values in the new x, y locations
new_values = []
for xi, yi in zip(x_new, y_new):
    srf.set_pos((xi, yi))
    field_value = srf()[0]
    new_values.append(field_value if field_value >= 0 else 0)

# Coordinates
# Name of columns
x_col = "X"
y_col = "Y"
field_col = "FIELD"
df_wells = pd.DataFrame({x_col: x_new, y_col: y_new, field_col: new_values})
#%% Create semivariogram
# Parameters of covariance model
dim = 2
var = 2
len_scale = 8
angles_tol = np.pi / 16
bandwidth = 8
# Position (Lat, long)
pos = [df_wells[x_col], df_wells[y_col]]
vals = df_wells[field_col]

# Estimate the variogram
# bins = range(0, 40, 2)
bin_center, dir_vario = gs.vario_estimate(
    pos,
    vals,
    direction=gs.rotated_main_axes(dim=dim, angles=angle),
    angles_tol=angles_tol,
    bandwidth=bandwidth,
)
#%%
# Fit the variogram with a covariance model
model_fit = gs.Exponential(dim=dim, angles=angle)
model_fit.fit_variogram(bin_center, dir_vario)
# Create conditioned Random Field
krige = gs.Krige(model_fit, cond_pos=pos, cond_val=new_values)
new_srf = gs.CondSRF(krige)
new_srf.set_pos((x, y), "structured")
new_srf()
#%% Plot semivariogram
fig_3, ax_3 = plt.subplots(1, 1, figsize=(8, 8))
ax_3.scatter(bin_center, dir_vario[0], label="Empirical semivariogram")
ax_3.scatter(bin_center, dir_vario[1], label="Empirical semivariogram 2")
model_fit.plot("vario_axis", axis=0, ax=ax_3, label='fit on axis 0')
model_fit.plot("vario_axis", axis=1, ax=ax_3, label="fit on axis 1")
plt.show()
#%%
# Plot the fitting model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(x=df_wells[x_col], y=df_wells[y_col], s=5, c="k")
srf.set_pos((x, y), "structured")
srf()
ax1_im = ax1.imshow(
    srf["field"].T,
    origin="lower",
    vmin=0,
    vmax=np.max(srf["field"].max()),
    extent=[0, 100, 0, 100],
    cmap="viridis",
)
ax1.set_title("Original Field")
ax2.scatter(x=df_wells[x_col], y=df_wells[y_col], s=5, c="k")
ax2_im = ax2.imshow(
    new_srf["field"].T,
    origin="lower",
    vmin=0,
    vmax=np.max(srf["field"].max()),
    extent=[0, 100, 0, 100],
    cmap="viridis",
)
ax2.set_title(f"Estimated Field Based on {n_wells} wells")
fig.show()

#%% Creating the field from the starting point again
new_srf_2 = gs.CondSRF(krige)
new_srf_2.set_pos((x, y), "structured")
#%% Create multiple random realizations
seed = gs.random.MasterRNG(4)
ens_no = 4
for i in range(ens_no):
    new_srf_2(seed=seed(), store=[f"fld{i}", False, False])
    
fig, ax = plt.subplots(ens_no + 1, ens_no + 1, figsize=(8, 8))
# plotting kwargs for scatter and image
vmax = np.max(new_srf_2.all_fields)
# Get the limits in x and y directions
xlim = (np.min(new_srf_2.pos[0]), np.max(new_srf_2.pos[0]))
ylim = (np.min(new_srf_2.pos[1]), np.max(new_srf_2.pos[1]))
sc_kw = dict(c=vals, edgecolors="k", vmin=0, vmax=vmax)
im_kw = dict(extent=[*xlim, *ylim], origin="lower", vmin=0, vmax=vmax)

for i in range(ens_no):
    # conditioned fields and conditions
    ax[i + 1, 0].imshow(new_srf_2[i].T, **im_kw)
    ax[i + 1, 0].scatter(*pos, **sc_kw)
    ax[i + 1, 0].set_ylabel(f"Field {i}", fontsize=10)
    ax[0, i + 1].imshow(new_srf_2[i].T, **im_kw)
    ax[0, i + 1].scatter(*pos, **sc_kw)
    ax[0, i + 1].set_title(f"Field {i}", fontsize=10)
    # absolute differences
    for j in range(ens_no):
        ax[i + 1, j + 1].imshow(np.abs(new_srf_2[i] - new_srf_2[j]).T, **im_kw)

# beautify plots
ax[0, 0].axis("off")
for a in ax.flatten():
    a.set_xticklabels([]), a.set_yticklabels([])
    a.set_xticks([]), a.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
fig.show()
#%% New wells estimation
well_1 = (30, 60)
well_2 = (50, 90)
well_3 = (50, 70)
well_4 = (70, 50)
well_5 = (30, 30)
#%% Loop over all the wells to extract the field value
wells = [well_1, well_2, well_3, well_4, well_5]
nw_dict = {f"well_{i}": well for i, well in enumerate(wells, start=1)}
n = 100  # Generate the conditioned random field "n"times and extract the values
# Make sure you use a seed so all the wells are created with the same random numbers
srf_nw = gs.CondSRF(krige)
well_results = {}
for well_name, new_well in nw_dict.items():
    # Set the position of conditioned random field to the new well
    srf_nw.set_pos((new_well[0], new_well[1]), 'structured')
    # Generate the "n" realizations
    for i in range(n):
        srf_nw(seed=seed(), store=[f"fld{i}", False, False])
    # Extract the field values from the "n" realizations
    well_prop = np.array([val[0][0] for val in srf_nw.all_fields])
    # Store the results in the results dictionary with each well name
    well_results[well_name] = well_prop


#%%
df_wellresults = pd.DataFrame(well_results)
df_wellresults.plot.kde()
plt.show()

#%% Plot New wells in field
fig_oc, ax_oc = plt.subplots(1, 1, figsize=(8, 8))
ax_oc.scatter(x=df_wells[x_col], y=df_wells[y_col], s=5, c="k")
ax_oc_im = ax_oc.imshow(
    new_srf["field"].T,
    origin="lower",
    vmin=0,
    vmax=np.max(srf["field"].max()),
    extent=[0, 100, 0, 100],
    cmap="viridis",
)
ax_oc.set_title(f"Estimated Field Based on {n_wells} wells")

for well_name, location in nw_dict.items():
    ax_oc.scatter(location[0], location[1], c="r", s=50)
    ax_oc.annotate(well_name, location)

fig_oc.show()
