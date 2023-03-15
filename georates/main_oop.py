from math import radians
import gstools as gs
from georates.model.model import Well, SyntheticField, WellGenerator, VariogramAnalysis, RandomField
import matplotlib.pyplot as plt
import numpy as np
#%% Create synthetic field
seed = 123
grid_resolution = 1
min_loc, max_loc = 0, 100
angle = radians(130)
mean = 10
model = gs.Exponential(dim=2, var=2, len_scale=[20, 8], angles=angle)

synth_field = SyntheticField(
    "auca", mean, seed, grid_resolution, (min_loc, max_loc), model, (0, None)
)
xy = synth_field.xy
print(xy)
#%%
fig, ax, axim = synth_field.plot_field(show_plot=False)
ax.set_title("Synthetic Field")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()
#%% Create synthetic wells
n_wells = 200
seed_nw = 28
well_generator = WellGenerator(synth_field, seed_nw)
wells = well_generator.generate_new_vertical_wells(n_wells)
#%%
print(wells)
#%% Define Variogram Analysis
vario_analysis = VariogramAnalysis(2, angle, np.pi / 16, 8, wells)
vario_analysis.plot_variogram(plot_model=False)
plt.show()
#%% Create covmodel
model_exp = gs.Exponential(dim=2, var=2, len_scale=[20, 8], angles=radians(180))
vario_analysis.covmodel = model_exp
vario_analysis.plot_variogram(plot_model=True)
plt.show()

#%%
random_field = RandomField(model_exp, wells, xy)
print(random_field)