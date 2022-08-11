import numpy as np
import gstools as gs
from typing import List, Tuple, Optional

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
    def __init__(
        self,
        field_name: str,
        mean: float,
        seed: int,
        grid_resolution: int,
        limits: Tuple[int, int],
        cov_model: gs.CovModel,
        trunc_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        """
        Create a synthetic field.

        Parameters
        ----------
        field_name
        mean
        seed
        grid_resolution
        limits
        cov_model
        """

        self.field_name = field_name
        self.seed = seed
        self.grid_resolution = grid_resolution
        self.limits = limits
        self.cov_model = cov_model
        self.mean = mean
        self.trunc_limits = trunc_limits
        # Private attributes
        self.__xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.field: Optional[np.ndarray] = None

    # getter / setter

    @property
    def _xy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__xy is None:
            self._generate_xy()
        return self.__xy

    @_xy.setter
    def _xy(self, value: Tuple[np.ndarray, np.ndarray]):
        raise ValueError("xy is a read-only attribute")

    def _generate_xy(self):
        x = y = np.arange(*self.limits, self.grid_resolution)
        self.__xy = x, y

    def generate_srf(self) -> np.ndarray:
        srf = gs.SRF(self.cov_model, mean=self.mean, seed=self.seed)
        field = srf.structured(self._xy)

        field_lower_limit, field_upper_limit = self.trunc_limits

        if field_lower_limit is not None:
            field = np.where(field < field_lower_limit, field_lower_limit, field)

        if field_upper_limit is not None:
            field = np.where(field > field_upper_limit, field_upper_limit, field)

        return field

    def plot_field(self):
        pass



