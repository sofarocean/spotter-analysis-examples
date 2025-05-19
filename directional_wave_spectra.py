"""

Maximum entropy method 2
========================
This module contains functions that implements the MEM2 method (see Kim1995 and references therein) and is used to
estimate the directional distribution that maximizes the entropy of the solution with entropy defined as
    $$\\int - D * log(D) \\, d\\theta$$

such that the resulting distribution $D(\theta)$ reproduces the observed moments. I.e.
    $$\\int  D \\, d\\theta= 1$$
    $$\\int \\cos(\\theta) D \\, d\\theta = a_1 $$
    $$\\int \\sin(\\theta) D \\, d\\theta = b_1 $$
    $$\\int \\cos(2\\theta) D \\, d\\theta = a_2$$
    $$\\int \\sin(2\\theta) D \\, d\\theta = b_2$$
and
    $$ D(\\theta) \\geq 0$$

References:

    Kim, T., Lin, L. H., & Wang, H. (1995). Application of maximum entropy method
    to the real sea data. In Coastal Engineering 1994 (pp. 340-355).

    link: https://icce-ojs-tamu.tdl.org/icce/index.php/icce/article/download/4967/4647
    (working as of May 29, 2022)

"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve

# Entry Function
# =============================================================================


def mem2(
    directions_radians: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """
    Estimate the directional distribution from the Fourier moments using the MEM2 method.

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

    :param a1: 1d array of cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b1: 1d array of sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param a2: 1d array of double angle cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b2: 1d array of double angle sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :return: array with shape [number_of_points, number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.


    """
    n_points, n_frequencies = a1.shape
    print(n_points, n_frequencies)
    directional_distribution = np.zeros((n_points, n_frequencies, len(directions_radians)))

    direction_increment_downward_difference = (directions_radians - np.roll(directions_radians, 1) + np.pi) % (2 * np.pi) - np.pi

    direction_increment_upward_difference = (-(directions_radians - np.roll(directions_radians, -1) + np.pi) % (2 * np.pi) - np.pi)

    direction_increment = (direction_increment_downward_difference + direction_increment_upward_difference) / 2

    # Calculate the needed Fourier transform twiddle factors to calculate moments.
    twiddle_factors = np.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = np.cos(directions_radians)
    twiddle_factors[1, :] = np.sin(directions_radians)
    twiddle_factors[2, :] = np.cos(2 * directions_radians)
    twiddle_factors[3, :] = np.sin(2 * directions_radians)

    guess = _initial_value(a1, b1, a2, b2)
    for ipoint in range(0, n_points):

        # Note; entries to directional_distribution[ipoint, :, :] is modified in the call below. This avoids creation
        # of memory for the resulting array at the expense of allowing for side effects.
        _mem2_newton_point(
            directional_distribution[ipoint, :, :],
            a1[ipoint, :],
            b1[ipoint, :],
            a2[ipoint, :],
            b2[ipoint, :],
            guess[ipoint, :, :],
            direction_increment,
            twiddle_factors,
        )

    return directional_distribution


# frequency iteration
# ----------------------

def _mem2_newton_point(
    out,
    a1,
    b1,
    a2,
    b2,
    guess,
    direction_increment,
    twiddle_factors,
):
    """

    :param out: a (view) of the array that will contain the output
    :param a1: 1d array of cosine directional moment as function of frequency,
    :param b1: 1d array of sine directional moment as function of frequency,
    :param a2: 1d array of double angle cosine directional moment as function of frequency,
    :param b2: 1d array of double angle sine directional moment as function of frequency,
    :param guess: initial guess of the lagrange multipliers
    :param direction_increment: directional stepsize used in the integration, nd-array
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :return: None - we use side-effects to pass the results back to the caller (modifying out)
    """
    number_of_frequencies = a1.shape[0]
    for ifreq in range(0, number_of_frequencies):
        #
        moments = np.array([a1[ifreq], b1[ifreq], a2[ifreq], b2[ifreq]])
        out[ifreq, :] = _mem2_newton_solver(
            moments,
            guess[ifreq, :],
            direction_increment,
            twiddle_factors,
        )


# mem2 numerical solver
# ----------------------

def _mem2_newton_solver(
    moments: np.ndarray,
    guess: np.ndarray,
    direction_increment: np.ndarray,
    twiddle_factors: np.ndarray,
) -> np.ndarray:
    max_iter = 100
    rcond = 1e-6
    atol = 0.01
    max_line_search_depth = 8

    if np.any(np.isnan(guess)):
        return np.zeros_like(direction_increment)

    current_iterate = guess
    current_func = _moment_constraints(current_iterate, twiddle_factors, moments, direction_increment)

    for iteration in range(max_iter):
        if np.linalg.norm(current_func) < atol:
            break

        jacobian = _mem2_jacobian(current_iterate, twiddle_factors, direction_increment)
        try:
            update = cho_solve(cho_factor(jacobian), -current_func)
        except Exception:
            update = np.linalg.lstsq(jacobian, -current_func, rcond=rcond)[0]

        if np.linalg.norm(update) == 0.0:
            break  # Stationary point

        for _ in range(max_line_search_depth):
            trial_iterate = current_iterate + update
            trial_func = _moment_constraints(trial_iterate, twiddle_factors, moments, direction_increment)

            if np.linalg.norm(trial_func) < np.linalg.norm(current_func):
                current_iterate = trial_iterate
                current_func = trial_func
                break
            update *= 0.5
        else:
            print("Line search failed")
            break

    return _mem2_directional_distribution(current_iterate, direction_increment, twiddle_factors)


# mem2 functions
# ----------------------

def _moment_constraints(lambdas, twiddle_factors, moments, direction_increment):
    """
    Construct the nonlinear equations we need to solve for lambda. The constraints are the difference between the
    desired moments a1,b1,a2,b2 and the moment calculated from the current distribution guess and for a perfect fit
    should be 0.

    To note: we differ from Kim et al here who formulate the constraints using un-normalized equations. Here we opt to
    use the normalized version as that allows us to cast the error / or mismatch directly in terms of an error in the
    moments.

    :param lambdas: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param moments: [a1,b1,a2,b2]
    :param direction_increment: directional stepsize used in the integration, nd-array
    :return: array (length=4) with the difference between desired moments and those calculated from the current
        approximate distribution
    """

    # Get the current estimate of the directional distribution
    dist = _mem2_directional_distribution(lambdas, direction_increment, twiddle_factors)
    out = np.zeros(4)
    for mm in range(0, 4):
        # note - the part after the "-" is just a discrete approximation of the Fourier sine/cosine amplitude (moment)
        out[mm] = moments[mm] - np.sum((twiddle_factors[mm, :]) * dist * direction_increment)

    return out

def _mem2_jacobian(lagrange_multiplier, twiddle_factors, direction_increment):
    """
    Calculate the jacobian of the constraint equations. The resulting jacobian is a square and positive definite matrix

    :param lambdas: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param direction_increment: directional stepsize used in the integration, nd-array

    :return: a 4 by 4 matrix that is the Jacobian of the constraint equations.
    """
    inner_product = np.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    # We subtract the minimum to ensure that the values in the exponent do not become too large. This amounts to
    # multiplying with a constant - which is fine since we normalize anyway. Effectively, this avoids overflow errors
    # (or infinities) - at the expense of underflowing (which is less of an issue).
    #
    inner_product = inner_product - np.min(inner_product)

    normalization = 1 / np.sum(np.exp(-inner_product) * direction_increment)
    shape = np.exp(-inner_product)

    normalization_derivative = np.zeros(4)
    for mm in range(0, 4):
        normalization_derivative[mm] = normalization * np.sum(twiddle_factors[mm, :] * np.exp(-inner_product) * direction_increment)

    # To note- we have to multiply separately to avoid potential underflow/overflow errors.
    normalization_derivative = normalization_derivative * normalization

    shape_derivative = np.zeros((4, twiddle_factors.shape[1]))
    for mm in range(0, 4):
        shape_derivative[mm, :] = -twiddle_factors[mm, :] * shape

    jacobian = np.zeros([4, 4])
    for mm in range(0, 4):
        # we make use of symmetry and only explicitly calculate up to the diagonal
        for nn in range(0, mm + 1):
            jacobian[mm, nn] = -np.sum(
                twiddle_factors[mm, :]
                * direction_increment
                * (
                    normalization * shape_derivative[nn, :]
                    + shape * normalization_derivative[nn]
                ),
                -1,
            )
            if nn != mm:
                jacobian[nn, mm] = jacobian[mm, nn]
    return jacobian

def _mem2_directional_distribution(
    lagrange_multiplier,
    direction_increment,
    twiddle_factors,
) -> np.ndarray:
    """
    Given the solution for the Lagrange multipliers- reconstruct the directional
    distribution.
    :param lagrange_multiplier: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param direction_increment: directional stepsize used in the integration, nd-array
    :return: Directional distribution arrays as a function of directions
    """
    inner_product = np.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    inner_product = inner_product - np.min(inner_product)

    normalization = 1 / np.sum(np.exp(-inner_product) * direction_increment)
    return np.exp(-inner_product) * normalization

def _initial_value(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray):
    """
    Initial guess of the Lagrange Multipliers according to the "MEM AP2" approximation
    found im Kim1995

    :param a1: moment a1
    :param b1: moment b1
    :param a2: moment a2
    :param b2: moment b2
    :return: initial guess of the lagrange multipliers, with the same leading dimensions as input.
    """
    guess = np.empty((*a1.shape, 4))
    fac = 1 + a1**2 + b1**2 + a2**2 + b2**2
    guess[..., 0] = 2 * a1 * a2 + 2 * b1 * b2 - 2 * a1 * fac
    guess[..., 1] = 2 * a1 * b2 - 2 * b1 * a2 - 2 * b1 * fac
    guess[..., 2] = a1**2 - b1**2 - 2 * a2 * fac
    guess[..., 3] = 2 * a1 * b1 - 2 * b2 * fac
    return guess

def _get_direction_increment(directions_radians: np.ndarray) -> np.ndarray:
    """
    calculate the stepsize used for midpoint integration. The directions
    represent the center of the interval - and we want to find the dimensions of
    the interval (difference between the preceding and successive midpoint).

    :param directions_radians: array of radian directions
    :return: array of radian intervals
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


"""
Estimate
======================
Main module for estimating directional spectra from directional moments. Core functionality is provided by the
`estimate_directional_spectrum_from_moments` function.
"""
def estimate_directional_spectrum(
    e: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Construct a 2D directional energy spectrum based on the directional moments and a specified spectral reconstruction
    method.

    :param e: nd array of variance/energy density as a function of frequency. The trailing dimension is assumed to be
    the frequency dimension.

    :param a1: nd array of cosine directional moment as function of frequency. The trailing dimension is assumed to be
    the frequency dimension.

    :param b1: nd array of sine directional moment as function of frequency. The trailing dimension is assumed to be
    the frequency dimension.

    :param a2: nd array of double angle cosine directional moment as function
    of frequency. The trailing dimension is assumed to be the frequency dimension.

    :param b2: nd array of double angle sine directional moment as function of
    frequency, The trailing dimension is assumed to be the frequency dimension.

    :param direction: 1d array of wave directions in radians. Directional convention is the same as associated with
    the Fourier moments (typically going to, anti-clockswise from east).

    :return: numpy.ndarray of shape (..., number_of_directions) containing the directional energy spectrum

    REFERENCES:
    Benoit, M. (1993). Practical comparative performance survey of methods
        used for estimating directional wave spectra from heave-pitch-roll data.
        In Coastal Engineering 1992 (pp. 62-75).

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
        directional distribution in ocean wave spectra.
        Journal of Physical Oceanography, 16(12), 2052-2060.
    """

    # convert degrees to radians
    direction_radians = direction * (np.pi / 180)

    output_shape = list(a1.shape) + [len(direction)]
    if a1.ndim == 1:
        input_shape = [1, a1.shape[-1]]
    else:
        input_shape = [np.prod(a1.shape[0:-1]), a1.shape[-1]]

    a1 = a1.reshape(input_shape)
    b1 = b1.reshape(input_shape)
    a2 = a2.reshape(input_shape)
    b2 = b2.reshape(input_shape)

    # compute the directional distribution using the maximum entropy method above
    res = mem2(direction_radians, a1, b1, a2, b2)

    return res.reshape(output_shape) * (np.pi / 180) * e[..., None]
