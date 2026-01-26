import numpy as np
from scipy.optimize import root
from numba import njit
from dataclasses import dataclass, field
from scipy.signal import welch, csd

## dataclass definitions

@dataclass
class Displacement:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: np.ndarray
    n: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class DirectionalMoments:
    a1: np.ndarray = field(default_factory=lambda: np.array([]))
    b1: np.ndarray = field(default_factory=lambda: np.array([]))
    a2: np.ndarray = field(default_factory=lambda: np.array([]))
    b2: np.ndarray = field(default_factory=lambda: np.array([]))

    def append(self, other: 'DirectionalMoments') -> None:
        def normalize_array(x: np.ndarray) -> np.ndarray:
            if x.ndim == 0:
                return np.array([x])
            if x.ndim == 1:
                return x.reshape(1, -1)
            return x

        def concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if a.size == 0:
                return normalize_array(b)
            if b.size == 0:
                return normalize_array(a)
            a_norm = normalize_array(a)
            b_norm = normalize_array(b)
            if a_norm.shape[1:] != b_norm.shape[1:]:
                raise ValueError(f"Incompatible shapes: {a.shape} vs {b.shape}")
            return np.concatenate((a_norm, b_norm), axis=0)

        self.a1 = concat(self.a1, other.a1)
        self.b1 = concat(self.b1, other.b1)
        self.a2 = concat(self.a2, other.a2)
        self.b2 = concat(self.b2, other.b2)

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
    enn: np.ndarray = field(default_factory=lambda: np.array([]))
    cxy: np.ndarray = field(default_factory=lambda: np.array([]))
    czn: np.ndarray = field(default_factory=lambda: np.array([]))
    qxz: np.ndarray = field(default_factory=lambda: np.array([]))
    qyz: np.ndarray = field(default_factory=lambda: np.array([]))
    directional_moments: DirectionalMoments = field(default_factory=DirectionalMoments)
    directional_spectra: np.ndarray = field(default_factory=lambda: np.array([]))

    def append(self, other: 'Spectrum') -> None:
        """Appends another Spectrum object to this one along the time dimension."""
        if not np.array_equal(self.frequency, other.frequency):
            raise ValueError("Frequencies must match to append spectra.")

        def normalize_array(x: np.ndarray, target_ndim: int) -> np.ndarray:
            """Ensure array is at least `target_ndim` by adding leading dims."""
            if np.isscalar(x) or x.ndim == 0:
                x = np.array([x])
            while x.ndim < target_ndim:
                x = np.expand_dims(x, axis=0)
            return x

        def concat(a: np.ndarray, b: np.ndarray, target_ndim: int) -> np.ndarray:
            """Concatenate a and b along time dimension (axis=0)."""
            if a.size == 0:
                return normalize_array(b, target_ndim)
            if b.size == 0:
                return normalize_array(a, target_ndim)

            a_norm = normalize_array(a, target_ndim)
            b_norm = normalize_array(b, target_ndim)

            if a_norm.shape[1:] != b_norm.shape[1:]:
                raise ValueError(f"Incompatible shapes for concatenation: {a.shape} vs {b.shape}")
            return np.concatenate((a_norm, b_norm), axis=0)

        # Spectral variables (1D or 2D → stack along time axis)
        self.ezz = concat(self.ezz, other.ezz, target_ndim=2)
        self.exx = concat(self.exx, other.exx, target_ndim=2)
        self.eyy = concat(self.eyy, other.eyy, target_ndim=2)
        self.enn = concat(self.enn, other.enn, target_ndim=2)
        self.cxy = concat(self.cxy, other.cxy, target_ndim=2)
        self.czn = concat(self.czn, other.czn, target_ndim=2)
        self.qxz = concat(self.qxz, other.qxz, target_ndim=2)
        self.qyz = concat(self.qyz, other.qyz, target_ndim=2)

        # Time (scalar or 1D → stacked as (T,))
        self.time = concat(self.time, other.time, target_ndim=1)

        # Directional spectra (2D or 3D → stacked along time axis)
        self.directional_spectra = concat(self.directional_spectra, other.directional_spectra, target_ndim=3)

        # Directional moments: assume it's a list of dicts or similar
        self.directional_moments.append(other.directional_moments)


def custom_hann(N): # from Pieter in slack thread
    n = np.linspace(0,1,N,endpoint=False)
    return np.sqrt( 8 / 3) * (0.5 - 0.5 * np.cos(n * np.pi*2))

def spectral_wave_analysis(
        disp: Displacement,
        fs: float = 2.5,
        N: int = 256,
        segment_length_seconds: int = 1800
) -> Spectrum:

    HANN_N = custom_hann(N)
    freqs, _ = welch(disp.x[:N], fs=fs, nperseg=N)

    n_samples = int(np.floor(len(disp.x) / fs / segment_length_seconds))
    sample_len = int(np.floor(len(disp.x) / n_samples))

    def init_spectrum_array():
        return np.zeros((n_samples, len(freqs)))

    Exx_pos, Eyy_pos, Ezz_pos, Enn_pos = init_spectrum_array(), init_spectrum_array(), init_spectrum_array(), init_spectrum_array()
    Cxy_pos, Czn_pos, Qxz_pos, Qyz_pos = init_spectrum_array(), init_spectrum_array(), init_spectrum_array(), init_spectrum_array()

    # reshape for easy looping

    x_arr = disp.x[0:n_samples * sample_len].reshape(n_samples, sample_len) # throws away the end of the time series that doesn't have N points
    y_arr = disp.y[0:n_samples * sample_len].reshape(n_samples, sample_len)
    z_arr = disp.z[0:n_samples * sample_len].reshape(n_samples, sample_len)

    if disp.n.size > 0:
        n_arr = disp.n[0:n_samples * sample_len].reshape(n_samples, sample_len)

    for t in range(n_samples):
        _, Exx = welch(x_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')
        _, Eyy = welch(y_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')
        _, Ezz = welch(z_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')

        _, Cxy = csd(x_arr[t], y_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')
        _, Qxz = csd(x_arr[t], z_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')
        _, Qyz = csd(y_arr[t], z_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')

        Exx_pos[t, :] = Exx
        Eyy_pos[t, :] = Eyy
        Ezz_pos[t, :] = Ezz
        Cxy_pos[t, :] = np.real(Cxy)
        Qxz_pos[t, :] = np.imag(Qxz)
        Qyz_pos[t, :] = np.imag(Qyz)

        if disp.n.size > 0:
            _, Enn = welch(n_arr[t], fs=fs, nperseg=N, noverlap=N//2, window=HANN_N, scaling='density')
            _, Czn = csd(z_arr[t], n_arr[t], fs=fs, nperseg=N, noverlap=N // 2, window=HANN_N, scaling='density')

            Enn_pos[t, :] = Enn
            Czn_pos[t, :] = np.real(Czn)

    spectrum = Spectrum(
        frequency=freqs,
        ezz=np.squeeze(Ezz_pos),
        eyy=np.squeeze(Eyy_pos),
        exx=np.squeeze(Exx_pos),
        enn=np.squeeze(Enn_pos),
        cxy=np.squeeze(Cxy_pos),
        czn=np.squeeze(Czn_pos),
        qxz=np.squeeze(Qxz_pos),
        qyz=np.squeeze(Qyz_pos),
    )

    return spectrum

# directional wave spectra

def calculate_directional_moments(spectrum: Spectrum) -> DirectionalMoments:
    directional_moments = DirectionalMoments(
        a1 = spectrum.qxz / np.sqrt(spectrum.ezz * (spectrum.exx + spectrum.eyy)),
        b1 = spectrum.qyz / np.sqrt(spectrum.ezz * (spectrum.exx + spectrum.eyy)),
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

    phi = (3 * np.pi / 2) - directions_radians  # uses meteorological directional convention

    twiddle_factors[0, :] = np.cos(phi)
    twiddle_factors[1, :] = np.sin(phi)
    twiddle_factors[2, :] = np.cos(2 * phi)
    twiddle_factors[3, :] = np.sin(2 * phi)  

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
        n_directions: int = 180,
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

    if spectrum.ezz.ndim == 1: #TODO tomorrow Anna, fix for multidimensional spectra
        spectrum.time = disp.time[-1]
    else:
        spectrum.time = disp.time[int(segment_length_seconds*fs):-1:int(segment_length_seconds*fs)]

    return spectrum

