import numpy as np
from dataclasses import dataclass, field
from wave_signal_processing.datatypes import DirectionalMoments, Spectrum
from typing import Union

@dataclass
class BulkParameters:
    time: np.ndarray = field(default_factory=lambda: np.array([]))
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
    def _is_null_ezz_slice(ezz_slice: np.ndarray) -> bool:
        vals = np.asarray(ezz_slice, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return True
        return np.allclose(finite, 0.0)

    def _row_or_full(arr: np.ndarray, idx: int) -> np.ndarray:
        a = np.asarray(arr)
        if a.size == 0:
            return np.array([])
        if a.ndim == 1:
            return a
        return a[idx]

    def _slice_spectrum(spec: Spectrum, idx: int) -> Spectrum:
        dm = DirectionalMoments(
            a1=_row_or_full(spec.directional_moments.a1, idx),
            b1=_row_or_full(spec.directional_moments.b1, idx),
            a2=_row_or_full(spec.directional_moments.a2, idx),
            b2=_row_or_full(spec.directional_moments.b2, idx),
        )
        time_arr = np.asarray(spec.time)
        if time_arr.size == 0:
            time_val = np.array([])
        elif time_arr.ndim == 0:
            time_val = time_arr
        else:
            time_val = time_arr[idx]

        return Spectrum(
            frequency=spec.frequency,
            ezz=_row_or_full(spec.ezz, idx),
            time=time_val,
            direction=np.asarray(spec.direction),
            exx=_row_or_full(spec.exx, idx),
            eyy=_row_or_full(spec.eyy, idx),
            enn=_row_or_full(spec.enn, idx),
            cxy=_row_or_full(spec.cxy, idx),
            czn=_row_or_full(spec.czn, idx),
            qxz=_row_or_full(spec.qxz, idx),
            qyz=_row_or_full(spec.qyz, idx),
            directional_moments=dm,
            directional_spectra=np.array([]),
        )

    ezz = np.asarray(spectrum.ezz)
    if ezz.ndim == 1:
        if _is_null_ezz_slice(ezz):
            nan_val = np.nan
            return BulkParameters(
                time=spectrum.time,
                significant_wave_height=nan_val,
                mean_period=nan_val,
                mean_direction=nan_val,
                mean_directional_spreading=nan_val,
                peak_frequency=nan_val,
                peak_period=nan_val,
                peak_direction=nan_val,
                peak_directional_spreading=nan_val,
            )
        return BulkParameters(
            time=spectrum.time,
            significant_wave_height=significant_wave_height(spectrum),
            mean_period=mean_period(spectrum),
            mean_direction=mean_direction(spectrum),
            mean_directional_spreading=mean_directional_spreading(spectrum),
            peak_frequency=peak_frequency(spectrum),
            peak_period=peak_period(spectrum),
            peak_direction=peak_direction(spectrum),
            peak_directional_spreading=peak_directional_spreading(spectrum),
        )

    n_rows = ezz.shape[0]
    nan_arr = np.full(n_rows, np.nan)
    hs = nan_arr.copy()
    tm = nan_arr.copy()
    mdir = nan_arr.copy()
    mspread = nan_arr.copy()
    pf = nan_arr.copy()
    pp = nan_arr.copy()
    pdir = nan_arr.copy()
    pspread = nan_arr.copy()

    for idx in range(n_rows):
        if _is_null_ezz_slice(ezz[idx]):
            continue
        row_spec = _slice_spectrum(spectrum, idx)
        hs[idx] = significant_wave_height(row_spec)
        tm[idx] = mean_period(row_spec)
        mdir[idx] = mean_direction(row_spec)
        mspread[idx] = mean_directional_spreading(row_spec)
        pf[idx] = peak_frequency(row_spec)
        pp[idx] = peak_period(row_spec)
        pdir[idx] = peak_direction(row_spec)
        pspread[idx] = peak_directional_spreading(row_spec)

    return BulkParameters(
        time=spectrum.time,
        significant_wave_height=hs,
        mean_period=tm,
        mean_direction=mdir,
        mean_directional_spreading=mspread,
        peak_frequency=pf,
        peak_period=pp,
        peak_direction=pdir,
        peak_directional_spreading=pspread,
    )
