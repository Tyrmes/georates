import pandas as pd
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from math import radians
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

