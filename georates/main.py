import pandas as pd
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from math import radians
# %% Define values
data = {
    "Well": [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
        "P15",
    ],
    "X": [20, 0, 100, 260, 400, 150, 425, 260, 360, 450, 530, 520, 600, 600, 600],
    "Y": [80, 600, 400, 230, 550, 200, 320, 520, 130, 0, 150, 430, 300, 550, 0],
    "h": [20, 19, 21, 25, 30, 23, 35, 23, 28, 32, 45, 56, 48, 60, 43],
}

# %% Create dataframe
df = pd.DataFrame(data)
df.head()

#%% Create semivariogram

# Name of columns
X = "X"
Y = "Y"
H = "h"
# Parameters of covariance model
dim = 2
var = 2
len_scale = 8
angles_tol = np.pi / 16
bandwidth = 8
angles = radians(180)

# Position (Lat, long)
pos = df[[X, Y]].values.T
# Values -> Thickness
field = df[H].values.T

# Estimate the variogram
bin_center, dir_vario = gs.vario_estimate(
    pos, field, angles=angles, angles_tol=angles_tol, bandwidth=bandwidth
)

# Fit the variogram with a covariance model
model = gs.Gaussian(dim=dim, angles=angles)
model.fit_variogram(bin_center, dir_vario)

# Plot the fitting model
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(bin_center, dir_vario, label="Empirical semivariogram")
model.plot(ax=ax)
ax.legend()
plt.show()
print(model)

