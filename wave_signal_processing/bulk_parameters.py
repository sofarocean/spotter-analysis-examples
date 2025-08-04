import numpy as np
from dataclasses import dataclass, field
from wave_signal_processing.spectral_analysis import Spectrum
from typing import Union

@dataclass
class BulkParameters:
    significant_wave_height: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_period: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_direction: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_directional_spreading: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_frequency: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_period: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_direction: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_directional_spreading: np.ndarray = field(default_factory=lambda: np.array([]))
    wind: np.ndarray = field(default_factory=lambda: np.array([]))

def moment(spectrum: Spectrum, order: int, lower_id: int, upper_id: int) -> Union[np.ndarray, float]:
    spectrum.ezz = np.nan_to_num(spectrum.ezz, nan=0) # convert nans to zeros prior to integrating
    if spectrum.ezz.ndim == 1:
        m = np.trapz(spectrum.ezz[lower_id: upper_id] * spectrum.frequency[lower_id: upper_id]**order, spectrum.frequency[lower_id: upper_id])
    elif spectrum.ezz.ndim == 2:
        m = np.trapz(spectrum.ezz[:, lower_id: upper_id] * spectrum.frequency[lower_id: upper_id]**order, spectrum.frequency[lower_id: upper_id], axis=1)

    return m

def significant_wave_height(spectrum: Spectrum) -> Union[np.ndarray, float]:
    return 4 * np.sqrt(moment(spectrum, 0, 3, 127))


def mean_period(spectrum: Spectrum):
    m0 = moment(spectrum, 0, 3, 127)
    m1 = moment(spectrum, 1, 3, 127)
    return m0 / m1

def mean_direction(spectrum: Spectrum) -> Union[np.ndarray, float]:
    if spectrum.ezz.ndim == 1:
        mean_axis = 0
    else:
        mean_axis = 1
    a1_bar = np.nanmean(spectrum.qxz, axis=mean_axis) / np.sqrt(np.nanmean(spectrum.ezz, axis=mean_axis) * (np.nanmean(spectrum.exx, axis=mean_axis) + np.nanmean(spectrum.eyy, axis=mean_axis)))
    b1_bar = np.nanmean(spectrum.qyz, axis=mean_axis) / np.sqrt(np.nanmean(spectrum.ezz, axis=mean_axis) * (np.nanmean(spectrum.exx, axis=mean_axis) + np.nanmean(spectrum.eyy, axis=mean_axis)))

    mean_dir = 270 - (180 / np.pi) * np.arctan2(b1_bar, a1_bar)
    if type(mean_dir) == np.ndarray:
        mean_dir[mean_dir < 0] = mean_dir[mean_dir < 0] + 360 # necessary to force direction to be between 0° and 360°
        mean_dir[mean_dir > 360] = mean_dir[mean_dir > 360] - 360
    elif type(mean_dir) == np.float64:
        if mean_dir < 0:
            mean_dir += 360
        elif mean_dir > 360:
            mean_dir -= 360

    return mean_dir

def mean_directional_spreading(spectrum: Spectrum) -> Union[np.ndarray, float]:
    if spectrum.ezz.ndim == 1:
        mean_axis = 0
    else:
        mean_axis = 1
    a1_bar = np.nanmean(spectrum.qxz, axis=mean_axis) / np.sqrt(np.nanmean(spectrum.ezz, axis=mean_axis) * (np.nanmean(spectrum.exx, axis=mean_axis) + np.nanmean(spectrum.eyy, axis=mean_axis)))
    b1_bar = np.nanmean(spectrum.qyz, axis=mean_axis) / np.sqrt(np.nanmean(spectrum.ezz, axis=mean_axis) * (np.nanmean(spectrum.exx, axis=mean_axis) + np.nanmean(spectrum.eyy, axis=mean_axis)))

    return 180 / np.pi * np.sqrt(2 * (1 - np.sqrt(a1_bar**2 + b1_bar**2)))

def peak_frequency(spectrum: Spectrum) -> Union[np.ndarray, float]:
    if spectrum.ezz.ndim == 1:
        mean_axis = 0
    else:
        mean_axis = 1
    peak_idx = np.nanargmax(spectrum.ezz, axis=mean_axis)

    return spectrum.frequency[peak_idx]

def peak_period(spectrum: Spectrum) -> Union[np.ndarray, float]:

    return 1 / peak_frequency(spectrum)

def peak_direction(spectrum: Spectrum):
    if spectrum.directional_moments.b1.ndim == 2:
        peak_idx = np.nanargmax(spectrum.ezz, axis=1)
        b1_p = spectrum.directional_moments.b1[np.arange(spectrum.directional_moments.b1.shape[0]), peak_idx]
        a1_p = spectrum.directional_moments.a1[np.arange(spectrum.directional_moments.a1.shape[0]), peak_idx]
    elif spectrum.directional_moments.b1.ndim == 1:
        peak_idx = np.nanargmax(spectrum.ezz)
        b1_p = spectrum.directional_moments.b1[peak_idx]
        a1_p = spectrum.directional_moments.a1[peak_idx]

    peak_dir = 270 - (180 / np.pi) * np.arctan2(b1_p,a1_p)

    if type(peak_dir) == np.ndarray:
        peak_dir[peak_dir < 0] = peak_dir[peak_dir < 0] + 360 # necessary to force direction to be between 0° and 360°
        peak_dir[peak_dir > 360] = peak_dir[peak_dir > 360] - 360
    elif type(peak_dir) == np.float64:
        if peak_dir < 0:
            peak_dir += 360
        elif peak_dir > 360:
            peak_dir -= 360

    return peak_dir

def peak_directional_spreading(spectrum: Spectrum) -> Union[np.ndarray, float]:
    if spectrum.directional_moments.b1.ndim == 2:
        peak_idx = np.nanargmax(spectrum.ezz, axis=1)
        b1_p = spectrum.directional_moments.b1[np.arange(spectrum.directional_moments.b1.shape[0]), peak_idx]
        a1_p = spectrum.directional_moments.a1[np.arange(spectrum.directional_moments.a1.shape[0]), peak_idx]
    elif spectrum.directional_moments.b1.ndim == 1:
        peak_idx = np.nanargmax(spectrum.ezz)
        b1_p = spectrum.directional_moments.b1[peak_idx]
        a1_p = spectrum.directional_moments.a1[peak_idx]

    return 180 / np.pi * np.sqrt(2 * (1 - np.sqrt(a1_p ** 2 + b1_p ** 2)))

def compute_bulk_parameters(spectrum: Spectrum) -> BulkParameters:

    return BulkParameters(
        significant_wave_height = significant_wave_height(spectrum),
        mean_period = mean_period(spectrum),
        mean_direction = mean_direction(spectrum),
        mean_directional_spreading = mean_directional_spreading(spectrum),
        peak_frequency = peak_frequency(spectrum),
        peak_period = peak_period(spectrum),
        peak_direction = peak_direction(spectrum),
        peak_directional_spreading = peak_directional_spreading(spectrum)
    )