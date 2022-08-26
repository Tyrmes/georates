from math import radians
import gstools as gs
from georates.model.model import Well, SyntheticField, WellGenerator
import matplotlib.pyplot as plt
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
#%%
fig, ax, axim = synth_field.plot_field(show_plot=False)
ax.set_title("Synthetic Field")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()
#%% Create synthetic wells
n_wells = 50
seed_nw = 28
well_generator = WellGenerator(synth_field, seed_nw)
wells = well_generator.generate_new_vertical_wells(n_wells)
