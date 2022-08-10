import numpy as np
import gstools as gs
from typing import List, Tuple

# %% Creation of synthetic field


# %% Create existing wells from synthetic field
# %% Create semivariogram from wells
# %% Fit covariance to wells
# %% Estimate values from new wells

# Why objective oriented programming?
# code is more readable and easier to maintain
# code is more reusable
# code is more maintainable
# You create objects for two reasons:
# Objects that represent data
# - More properties
# Objects that represent behavior
# - More methods

# Class
# - Properties
# - Methods
# And instance of a class is an object

# Static type checker
# - Mypy

# Create a well class


class Well:
    def __init__(self, well_name: str, location: Tuple[float, float]):
        self.well_name = well_name
        self.location = location

    def start_producing(self):
        print(f"{self.well_name} is producing")


# %%

class SyntheticField:
    def __init__(self,
                 field_name: str,
                 seed: int,
                 grid_resolution: int,
                 limits: Tuple[int, int],
                 cov_model: gs.CovModel):
        """
        Create a synthetic field.

        :param field_name:
        :param seed:
        :param grid_resolution:
        :param limits:
        :param cov_model:

        """
        self.field_name = field_name
        self.seed = seed
        self.grid_resolution = grid_resolution
        self.limits = limits
        self.cov_model = cov_model

    def generate_points(self):
        model
        srf

    def create_model(self):
        self.model = gs.Exponential(dim=2, var=2, len_scale=[20, 8], angles=angle)
        srf = gs.SRF(model, mean=10, seed=seed)
        field = srf.structured([x, y])
        field = np.where(field < 0, 0, field)
        return field

# %%
class SynthField:
    def __int__(self, field_name: str, seed: int, grid_resolution: int,
                limits: Tuple[int, int], angle: float):
        self.field_name = field_name
        self.seed = seed
        self.grid_resolution = grid_resolution
        self.limits = limits
        self.angle = angle

    def model(self):
        model = gs.Exponential(dim=2, var=2, len_scale=[20, 8], angle= self.angle)


