import numpy as np
from scipy.optimize import root
from numba import njit
from dataclasses import dataclass, field

## dataclass definitions

@dataclass
class Displacement:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: np.ndarray

@dataclass
class DirectionalMoments:
    a1: np.ndarray = field(default_factory=lambda: np.array([]))
    b1: np.ndarray = field(default_factory=lambda: np.array([]))
    a2: np.ndarray = field(default_factory=lambda: np.array([]))
    b2: np.ndarray = field(default_factory=lambda: np.array([]))

    def reshape_for_solver(self) -> "DirectionalMoments":
        """
        Reshapes all directional moment arrays to 2D shape [n_points, n_frequencies]
        """
        def shape_moment(arr):
            if arr.ndim == 1:
                return arr[None, :]  # Add leading axis
            elif arr.ndim == 2:
                return arr
            else:
                # Flatten leading dims, preserve frequency axis
                return arr.reshape(-1, arr.shape[-1])

        return DirectionalMoments(
            a1=shape_moment(self.a1),
            b1=shape_moment(self.b1),
            a2=shape_moment(self.a2),
            b2=shape_moment(self.b2),
        )

@dataclass
class Spectrum:
    frequency: np.ndarray
    ezz: np.ndarray
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    exx: np.ndarray = field(default_factory=lambda: np.array([]))
    eyy: np.ndarray = field(default_factory=lambda: np.array([]))
    cxy: np.ndarray = field(default_factory=lambda: np.array([]))
    qzx: np.ndarray = field(default_factory=lambda: np.array([]))
    qzy: np.ndarray = field(default_factory=lambda: np.array([]))
    directional_moments: DirectionalMoments = field(default_factory=DirectionalMoments)
    directional_spectra: np.ndarray = field(default_factory=lambda: np.array([]))

## spectral helper functions
def window_time_series(time_series, window):
    rms_gain = np.sqrt((window**2).mean())
    ts_win = time_series * window
    return ts_win / rms_gain

def compute_cross_spectra(time_series1, time_series2):
    ts1fft = np.fft.rfft(time_series1)
    ts2fft = np.fft.rfft(time_series2)

    return ts1fft * ts2fft.conj()

## primary spectral analysis function

def spectral_wave_analysis(
        disp: Displacement,
        fs: float = 2.5,
        N: int = 256,
        segment_length_seconds: int = 1800
) -> Spectrum:

    window = np.hanning(N)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    n_samples = int(np.floor(len(disp.x) / fs / segment_length_seconds))
    sample_len = int(np.floor(len(disp.x) / n_samples))

    def init_spectrum_array():
        return np.zeros((n_samples, len(freqs)))

    Exx_pos, Eyy_pos, Ezz_pos = init_spectrum_array(), init_spectrum_array(), init_spectrum_array()
    Cxy_pos, Qzx_pos, Qzy_pos = init_spectrum_array(), init_spectrum_array(), init_spectrum_array()

    # reshape for easy looping

    x_arr = disp.x[0:n_samples * sample_len].reshape(n_samples, sample_len) # throws away the end of the time series that doesn't have N points
    y_arr = disp.y[0:n_samples * sample_len].reshape(n_samples, sample_len)
    z_arr = disp.z[0:n_samples * sample_len].reshape(n_samples, sample_len)

    for t in range(n_samples):
        Exx, Eyy, Ezz = [], [], []
        Cxy, Qzx, Qzy = [], [], []
        for i in range(0, sample_len - N + 1, N // 2):
            try:
                x_win = window_time_series(x_arr[t, i:i + 256], window)
                y_win = window_time_series(y_arr[t, i:i + 256], window)
                z_win = window_time_series(z_arr[t, i:i + 256], window)

                Exx.append(compute_cross_spectra(x_win, x_win) / N)
                Eyy.append(compute_cross_spectra(y_win, y_win) / N)
                Ezz.append(compute_cross_spectra(z_win, z_win) / N)

                Cxy.append(compute_cross_spectra(x_win, y_win) / N)
                Qzx.append(compute_cross_spectra(z_win, x_win) / N)
                Qzy.append(compute_cross_spectra(z_win, y_win) / N)

            except Exception as e:
                continue

        def finalize_spectrum(arr_list):
            arr = np.array(arr_list)
            return np.mean(np.real(arr), axis=0)

        Exx_pos[t, :] = finalize_spectrum(Exx)
        Eyy_pos[t, :] = finalize_spectrum(Eyy)
        Ezz_pos[t, :] = finalize_spectrum(Ezz)
        Cxy_pos[t, :] = finalize_spectrum(Cxy)
        Qzx_pos[t, :] = np.mean(np.imag(Qzx), axis=0)
        Qzy_pos[t, :] = np.mean(np.imag(Qzy), axis=0)

    spectrum = Spectrum(
        frequency=freqs,
        ezz=Ezz_pos,
        eyy=Eyy_pos,
        exx=Exx_pos,
        cxy=Cxy_pos,
        qzx=Qzx_pos,
        qzy=Qzy_pos,
    )
    return spectrum

# directional wave spectra

def calculate_directional_moments(spectrum: Spectrum) -> DirectionalMoments:
    directional_moments = DirectionalMoments(
        a1 = spectrum.qzx / np.sqrt(spectrum.ezz * (spectrum.exx + spectrum.eyy)),
        b1 = spectrum.qzy / np.sqrt(spectrum.ezz * (spectrum.exx + spectrum.eyy)),
        a2 = (spectrum.exx - spectrum.eyy) / (spectrum.exx + spectrum.eyy),
        b2 = 2 * spectrum.cxy / (spectrum.exx + spectrum.eyy),
    )

    return directional_moments

@njit(fastmath=True)
def mem2_directional_distribution(
    lambdas: np.ndarray,
    direction_increment: np.ndarray,
    twiddle_factors: np.ndarray,
) -> np.ndarray:
    """
    Given the solution for the Lagrange multipliers- reconstruct the directional
    distribution.
    :param lambdas: the Lagrange multipliers used to find the local minima of the error function
    :param twiddle_factors: [sin theta, cos theta, sin 2*theta, cos 2*theta] as a 4 by ndir array -- directional basis functions
    :param direction_increment: directional stepsize used in the integration, nd-array
    :return: Directional distribution array as a function of directions
    """
    inner_product = np.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lambdas[jj] * twiddle_factors[jj, :]

    inner_product = inner_product - np.min(inner_product)

    normalization = 1 / np.sum(np.exp(-inner_product) * direction_increment)
    return np.exp(-inner_product) * normalization

def get_direction_increment(directions_radians: np.ndarray) -> np.ndarray:
    """
    :param directions_radians: direction array (in radians)
    :return: directional increment for integrating (in radians)
    """
    # Calculate the forward difference appending the first entry to the back
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    forward_diff = (np.diff(directions_radians, append=directions_radians[0]) + np.pi) % (2 * np.pi) - np.pi

    # Calculate the backward difference prepending the last entry to the front
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    backward_diff = (np.diff(directions_radians, prepend=directions_radians[-1]) + np.pi) % (2 * np.pi) - np.pi

    # The interval we are interested in is the average of the forward and backward
    # differences.
    return (forward_diff + backward_diff) / 2

def initial_value(moments: DirectionalMoments) -> np.ndarray:
    """
     Initializes guess for root finding using the directional moments
    :param moments: directional moments; a1, b1, a2, b2
    :return guess: initial guess for root finding using the directional moments
    """
    guess = np.empty((*moments.a1.shape, 4))
    fac = 1 + moments.a1**2 + moments.b1**2 + moments.a2**2 + moments.b2**2
    guess[..., 0] = 2 * moments.a1 * moments.a2 + 2 * moments.b1 * moments.b2 - 2 * moments.a1 * fac
    guess[..., 1] = 2 * moments.a1 * moments.b2 - 2 * moments.b1 * moments.a2 - 2 * moments.b1 * fac
    guess[..., 2] = moments.a1**2 - moments.b1**2 - 2 * moments.a2 * fac
    guess[..., 3] = 2 * moments.a1 * moments.b1 - 2 * moments.b2 * fac
    return guess

@njit(fastmath=True)
def moment_constraints(lambdas: np.ndarray, twiddle_factors: np.ndarray, moments: np.ndarray, direction_increment: np.ndarray) -> np.ndarray:
    """
    This is the error function between the true directional moments and the discrete approximation. This is the function that we minimize (in root_finder below) in order to maximize entropy.
    :param lambdas: Lagrange multipliers used to find the local minima of the error function
    :param twiddle_factors: directional basis functions used to find the local minima of the error function
    :param moments: directional moments; a1, b1, a2, b2
    :param direction_increment: directional stepsize used in the integration
    :return: error between estimated and true directional moments
    """
    # Get the current estimate of the directional distribution
    dist = mem2_directional_distribution(lambdas, direction_increment, twiddle_factors)
    out = np.zeros(4)
    for mm in range(0, 4):
        # note - the part after the "-" is just a discrete approximation of the Fourier sine/cosine amplitude (moment)
        out[mm] = moments[mm] - np.sum((twiddle_factors[mm, :]) * dist * direction_increment)

    return out

def root_finder(
    directions_radians: np.ndarray,
    moments: DirectionalMoments,
) -> np.ndarray:
    """
    :param directions_radians: directional array (in radians)
    :param moments: directional moments; a1, b1, a2, b2
    :return: directional distribution array as a function of directions
    """

    number_of_frequencies = moments.a1.shape[-1]
    number_of_points = moments.a1.shape[0]

    directional_distribution = np.zeros((number_of_points, number_of_frequencies, len(directions_radians)))

    direction_increment = get_direction_increment(directions_radians)

    twiddle_factors = np.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = np.cos(directions_radians)
    twiddle_factors[1, :] = np.sin(directions_radians)
    twiddle_factors[2, :] = np.cos(2 * directions_radians)
    twiddle_factors[3, :] = np.sin(2 * directions_radians)

    guess = initial_value(moments)
    for ipoint in range(0, number_of_points):
        for ifreq in range(0, number_of_frequencies):

            moments_array = np.array(
                [
                    moments.a1[ipoint, ifreq],
                    moments.b1[ipoint, ifreq],
                    moments.a2[ipoint, ifreq],
                    moments.b2[ipoint, ifreq],
                ]
            )

            if np.any(np.isnan(guess[ipoint, ifreq, :])):
                continue

            res = root(
                moment_constraints,
                guess[ipoint, ifreq, :],
                args=(twiddle_factors, moments_array, direction_increment),
                method="lm",
            )
            lambdas = res.x

            directional_distribution[ipoint, ifreq, :] = mem2_directional_distribution(lambdas, direction_increment, twiddle_factors)

    return directional_distribution


"""
Estimate
======================
Main module for estimating directional spectra from directional moments. Core functionality is provided by the
`estimate_directional_spectrum` function.
"""
def estimate_directional_spectrum(
    ezz: np.ndarray,
    directional_moments: DirectionalMoments,
    direction: np.ndarray,
) -> np.ndarray:
    """
    :param ezz: variance density
    :param directional_moments: directional moments; a1, b1, a2, b2
    :param direction: direction vector (in degrees)

    :return: estimated 2D directional spectrum
    """
    # convert degrees to radians
    direction_radians = np.deg2rad(direction)

    output_shape = list(directional_moments.a1.shape) + [len(direction)]

    reshaped_moments = directional_moments.reshape_for_solver()
    D = root_finder(direction_radians, reshaped_moments)

    return D.reshape(output_shape) * (np.pi / 180) * ezz[..., None]

def compute_full_spectra_from_displacement(
        disp: Displacement,
        fs: float = 2.5,
        N: int = 256,
        segment_length_seconds: int = 1800,
        n_directions: int = 180
) -> Spectrum:
    """
    :param disp: Displacement; x, y, z, time
    :param fs: sampling frequency
    :param N: number of samples over which to compute a single spectrum
    :param segment_length_seconds: number of seconds over which to compute an average spectrum
    :param n_directions: number of directions for computing directional spectrum
    :return: Spectrum
    """

    spectrum = spectral_wave_analysis(disp, fs = fs, N = N, segment_length_seconds=segment_length_seconds)

    spectrum.directional_moments = calculate_directional_moments(spectrum)

    spectrum.direction = np.linspace(0, 360, n_directions, endpoint=True)

    spectrum.directional_spectra = estimate_directional_spectrum(spectrum.ezz,
                                                spectrum.directional_moments,
                                                spectrum.direction)

    spectrum.time = disp.time[range((int(segment_length_seconds * fs)), len(disp.time), int(segment_length_seconds * fs))]

    return spectrum

