import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import AxesImage
import gstools as gs
from typing import List, Tuple, Optional, Any, Union, Callable

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
from pandas import DataFrame


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

    def generate_new_vertical_wells(self, n_wells: int) -> list[Well]:
        rng = np.random.default_rng(self.seed)
        min_loc, max_loc = self.field.limits

        x_new = rng.integers(min_loc, max_loc, n_wells)
        y_new = rng.integers(min_loc, max_loc, n_wells)
        new_wells = []

        # wells_data = {'well_name': [], 'x_coord': [], 'y_coord': [], 'petro_value': []}

        for i, (xi, yi) in enumerate(zip(x_new, y_new)):
            field_value = self.field.get_field_value_with_position((xi, yi))
            field_value = field_value if field_value >= 0 else 0
            well_name = f"well_{i}"

            # wells_data['well_name'].append(well_name)
            # wells_data['x_coord'].append(xi)
            # wells_data['y_coord'].append(yi)
            # wells_data['petro_value'].append(field_value)

            well = Well(well_name, (xi, yi))
            well.petro_value = field_value
            new_wells.append(well)
        # new_wells_df = pd.DataFrame(wells_data)

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

    #
    # def well_features(self) -> pd.DataFrame:
    #     df_wells = pd.DataFrame({
    #         'x_coord': [well.location[0] for well in self.wells],
    #         'y_coord': [well.location[1] for well in self.wells],
    #         'petro_value': [well.petro_value for well in self.wells]
    #     })
    #     return df_wells

    def well_features(self) -> tuple[list[list[float]], list[Optional[float]]]:
        x = []
        y = []
        petro_value = []

        [x.append(well.location[0]) for well in self.wells],
        [y.append(well.location[1]) for well in self.wells],
        [petro_value.append(well.petro_value) for well in self.wells],
        pos = [x, y]

        return pos, petro_value

    def _calculate_variogram(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, petro_value = self.well_features()
        bin_center, dir_vario = gs.vario_estimate(
            pos,
            petro_value,
            direction=gs.rotated_main_axes(dim=self.dim, angles=self.angle),
            angles_tol=self.angles_tol,
            bandwidth=self.bandwidth
        )
        return bin_center, dir_vario

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

    def plot_variogram(self, plot_model=False, show_plot=False):
        bin_center, dir_vario = self._var_results
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(8, 8))
        ax_1.scatter(bin_center, dir_vario[0], label="Empirical semivariogram")
        ax_1.scatter(bin_center, dir_vario[1], label="Empirical semivariogram 2")
        # Plot covariance model
        if plot_model:
            self.covmodel.plot("vario_axis", axis=0, ax=ax_1, label="fit on axis 0")
            self.covmodel.plot("vario_axis", axis=1, ax=ax_1, label="fit on axis 1")
        if show_plot:
            fig_1.show()

    def fit_covmodel(self, covmodel: Optional[gs.CovModel] = None):
        if covmodel is not None:
            self.covmodel = covmodel
        self.covmodel.fit_variogram(*self._var_results)


# %%
# class WellProperties:
#     def __int__(self, well_list: list[Well]):
#         self.well_list = well_list
#
#     def well_position(self) -> list[float]:
#         well_position = list(self.well_list[0].location)
#         return well_position
#
#     def property_values(self):
#         property_values = self.well_list[1].petro_value
#         return property_values

class RandomField:
    def __init__(
            self,
            vario_fit: VariogramAnalysis.fit_covmodel,
            well_properties: VariogramAnalysis.well_features
    ):
        # Call methods from VariogramAnalysis
        self.vario_fit = vario_fit
        self.well_properties = well_properties

    @property
    def _krige(self) -> gs.krige.Krige:
        pos, petro_value = self.well_properties
        # if self.__crf is None:
        krige = gs.Krige(self.vario_fit, cond_pos= pos,
                         cond_val=petro_value)
        return krige

    def generate_crf(self) -> gs.CondSRF:
        pos, petro_value = self.well_properties
        new_crf = gs.CondSRF(self._krige)
        new_crf.set_pos(pos, 'structured')
        return new_crf

# %%

# cond_pos se refiere a crear

# def create_random_field(self):
#     wells = self.new_wells
#     position = wells
#
#
#
#
#     #krige = gs.Krige(model, cond_pos=pos, cond_val=new_values)
#     #new_srf = gs.CondSRF(krige)
#     #new_srf.set_pos((x, y), "structured")
#     #new_srf()
#
