import pandas as pd
import numpy as np
import gstools as gs

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
DIM = 2
VAR = 2
LEN_SCALE = 8
ANGLES_TOL = np.pi / 16
BANDWIDTH = 8

# Position (Lat, long)
pos = df[[X, Y]].values.T
# Values -> Thickness
field = df[H].values.T

# Estimate the variogram
emp_v = gs.vario_estimate(pos, field, latlon=True)

# Fit the variogram with a covariance model
model = gs.Exponential(dim=DIM, var=VAR, len_scale=LEN_SCALE)
model.fit_variogram(*emp_v, sill=np.var(field), nugget=True)

# Plot the fitting model
ax = model.plot()
ax.scatter(*emp_v, label="Empirical semivariogram")
ax.legend()
print(model)

