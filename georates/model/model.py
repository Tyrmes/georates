import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
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
        self.petro_value: Optional[float] = None

    def start_producing(self):
        print(f"{self.well_name} is producing")


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
            Name of the field.
        mean
            Mean of the field to be used in the generation of the field.
        seed
            Seed for the random number generator.
        grid_resolution
            Resolution of the grid
        limits
            Lower and upper (x, y) limits of the field grid
        cov_model
            Covariance model to be used in the generation of the field.
        trunc_limits
            Lower and upper limits of the field properties values.
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
        self.__srf: Optional[gs.SRF] = None
        self._field: Optional[np.ndarray] = None

    @property
    def _xy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__xy is None:
            self.__xy = self._generate_xy()
        return self.__xy

    @property
    def field(self) -> np.ndarray:
        if self._field is None:
            self._field = self._generate_srf()
        return self._field

    @property
    def _srf(self) -> gs.SRF:
        if self.__srf is None:
            self.__srf = gs.SRF(self.cov_model, mean=self.mean, seed=self.seed)
        return self.__srf

    def _generate_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        x = y = np.arange(*self.limits, self.grid_resolution)
        return x, y

    def _generate_srf(self) -> np.ndarray:
        field = self._srf.structured(self._xy)

        field_lower_limit, field_upper_limit = self.trunc_limits

        if field_lower_limit is not None:
            field = np.where(field < field_lower_limit, field_lower_limit, field)

        if field_upper_limit is not None:
            field = np.where(field > field_upper_limit, field_upper_limit, field)

        return field

    def plot_field(
        self, figsize=(8, 8), cmap="viridis", show_plot=True
    ) -> Tuple[plt.Figure, plt.Axes, AxesImage]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax_im = ax.imshow(
            self.field.T,
            origin="lower",
            vmin=self.trunc_limits[0],
            vmax=self.trunc_limits[1],
            extent=[*self.limits, *self.limits],
            cmap=cmap,
        )

        if show_plot:
            fig.show()

        return fig, ax, ax_im

    def get_field_value_with_position(self, position: Tuple[float, float]) -> float:
        self._srf.set_pos(position)
        return self._srf()[0]


class WellGenerator:
    def __init__(self, field: SyntheticField, seed: int):
        self.field = field
        self.seed = seed

    def generate_new_vertical_wells(self, n_wells: int) -> List[Well]:
        rng = np.random.default_rng(self.seed)
        min_loc, max_loc = self.field.limits

        x_new = rng.integers(min_loc, max_loc, n_wells)
        y_new = rng.integers(min_loc, max_loc, n_wells)

        new_wells = []
        for i, (xi, yi) in enumerate(zip(x_new, y_new)):
            field_value = self.field.get_field_value_with_position((xi, yi))
            field_value = field_value if field_value >= 0 else 0
            well_name = f"well_{i}"
            well = Well(well_name, (xi, yi))
            well.petro_value = field_value
            new_wells.append(well)

        return new_wells


class VariogramAnalysis:
    def __init__(
            self,
            dim: int,
            angle: float,
            angles_tol: float,
            bandwidth: int,
            wells: List[Well]
    ):
        self.dim = dim
        self.angle = angle
        self.angles_tol = angles_tol
        self.bandwidth = bandwidth
        self.wells = wells
        self.__var_results: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._covmodel: Optional[gs.CovModel] = None

    @property
    def covmodel(self) -> gs.CovModel:
        if self._covmodel is None:
            self._covmodel = self._default_cov_model()
            self._covmodel.dim = self.dim
            self._covmodel.angle = np.array([self.angle])
            return self._covmodel
        else:
            return self._covmodel

    @covmodel.setter
    def covmodel(self, covmodel: gs.CovModel):
        self._covmodel = covmodel
        self._covmodel.dim = self.dim
        self._covmodel.angle = np.array([self.angle])

    def _default_cov_model(self):
        model = gs.Exponential(self.dim, self.angle)
        return model

    @property
    def _var_results(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__var_results is None:
            self.__var_results = self._calculate_variogram()
        return self.__var_results

    def _calculate_variogram(self) -> Tuple[np.ndarray, np.ndarray]:
        x = []
        y = []
        vals = []
        for well in self.wells:
            x.append(well.location[0])
            y.append(well.location[1])
            vals.append(well.petro_value)

        pos = [x, y]

        bin_center, dir_vario = gs.vario_estimate(
            pos,
            vals,
            direction=gs.rotated_main_axes(dim=self.dim, angles=self.angle),
            angles_tol=self.angles_tol,
            bandwidth=self.bandwidth
        )

        return bin_center, dir_vario

    def plot_variogram(self, plot_model=False, show_plot=False, fit_model=False):
        bin_center, dir_vario = self._var_results
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(8, 8))
        ax_1.scatter(bin_center, dir_vario[0], label="Empirical semivariogram")
        ax_1.scatter(bin_center, dir_vario[1], label="Empirical semivariogram 2")
        # Plot covariance model
        if plot_model:
            self.covmodel.plot("vario_axis", axis=0, ax=ax_1, label="fit on axis 0")
            self.covmodel.plot("vario_axis", axis=1, ax=ax_1, label="fit on axis 1")
        elif fit_model:
            model_fit = self.covmodel
            model_fit.fit_variogram(bin_center, dir_vario)
            model_fit.plot("vario_axis", axis=0, ax=ax_1, label="fit on axis 0")
            model_fit.plot("vario_axis", axis=1, ax=ax_1, label="fit on axis 1")
        if show_plot:
            fig_1.show()

















