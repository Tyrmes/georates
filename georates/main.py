import pandas as pd
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from math import radians
# %% Define values
# Generate random values
x = np.random.RandomState(123).rand(100) * 100
y = np.random.RandomState(223).rand(100) * 100
model = gs.Exponential(dim=2, var=2, len_scale=[10, 8], angles=[radians(130), radians(70)])
srf = gs.SRF(model, mean=0, seed=123)
# Thickness
field = srf((x, y))
field = np.where(field < 0, 0, field)
# Coordinates
df = pd.DataFrame({'X': x, 'Y': y})

#%% Create semivariogram

# Name of columns
X = "X"
Y = "Y"

# Parameters of covariance model
dim = 2
var = 2
len_scale = 8
angles_tol = np.pi / 16
bandwidth = 8
angles = [radians(130), radians(70)]

# Position (Lat, long)
pos = df[[X, Y]].values.T
# x = df[X].values.T
# y = df[Y].values.T
# Values -> Thickness
field = field.T

# Estimate the variogram
bins = range(0, 40, 2)
bin_center, dir_vario = gs.vario_estimate(
    pos, field, bins, direction=gs.rotated_main_axes(dim=dim, angles=angles),
    angles_tol=angles_tol, bandwidth=bandwidth)

# Fit the variogram with a covariance model
#model = gs.Gaussian(dim=dim, angles=angles)
model.fit_variogram(bin_center, dir_vario)

# Plot the fitting model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(bin_center, dir_vario[0], label="Empirical semivariogram")
ax1.scatter(bin_center, dir_vario[1], label="Empirical semivariogram 2")
model.plot("vario_axis", axis=0, ax=ax1, label='fit on axis=0')
model.plot("vario_axis", axis=1, ax=ax1, label="fit on axis 1")
ax1.legend()
srf.plot(ax=ax2)
plt.show()
#print(model)

