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
from scipy.optimize import root
from numba import njit
import typing

# Entry Function
# =============================================================================

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

def initial_value(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
     Initializes guess for root finding using the directional moments
    :param a1: first cosine directional moment
    :param b1: first sine directional moment
    :param a2: second cosine directional moment
    :param b2: second sine directional moment
    :return guess: initial guess for root finding using the directional moments
    """
    guess = np.empty((*a1.shape, 4))
    fac = 1 + a1**2 + b1**2 + a2**2 + b2**2
    guess[..., 0] = 2 * a1 * a2 + 2 * b1 * b2 - 2 * a1 * fac
    guess[..., 1] = 2 * a1 * b2 - 2 * b1 * a2 - 2 * b1 * fac
    guess[..., 2] = a1**2 - b1**2 - 2 * a2 * fac
    guess[..., 3] = 2 * a1 * b1 - 2 * b2 * fac
    return guess

@njit(fastmath=True)
def moment_constraints(lambdas: np.ndarray, twiddle_factors: np.ndarray, moments: np.ndarray, direction_increment: np.ndarray) -> np.ndarray:
    """
    This is the error function between the true directional moments and the discrete approximation. This is the function that we minimize (in root_finder below) in order to maximize entropy.
    :param lambdas: Lagrange multipliers used to find the local minima of the error function
    :param twiddle_factors: directional basis functions used to find the local minima of the error function
    :param moments: directional moments
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
    a1: typing.Union[np.ndarray, float],
    b1: typing.Union[np.ndarray, float],
    a2: typing.Union[np.ndarray, float],
    b2: typing.Union[np.ndarray, float],
) -> np.ndarray:
    """
    :param directions_radians: directional array (in radians)
    :param a1: first cosine directional moment
    :param b1: first sine directional moment
    :param a2: second cosine directional moment
    :param b2: second sine directional moment

    :return: directional distribution array as a function of directions
    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = np.zeros((number_of_points, number_of_frequencies, len(directions_radians)))

    direction_increment = get_direction_increment(directions_radians)

    twiddle_factors = np.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = np.cos(directions_radians)
    twiddle_factors[1, :] = np.sin(directions_radians)
    twiddle_factors[2, :] = np.cos(2 * directions_radians)
    twiddle_factors[3, :] = np.sin(2 * directions_radians)

    guess = initial_value(a1, b1, a2, b2)
    for ipoint in range(0, number_of_points):
        for ifreq in range(0, number_of_frequencies):

            moments = np.array(
                [
                    a1[ipoint, ifreq],
                    b1[ipoint, ifreq],
                    a2[ipoint, ifreq],
                    b2[ipoint, ifreq],
                ]
            )

            if np.any(np.isnan(guess[ipoint, ifreq, :])):
                continue

            res = root(
                moment_constraints,
                guess[ipoint, ifreq, :],
                args=(twiddle_factors, moments, direction_increment),
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
    e: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    :param e: variance density
    :param a1: first cosine directional moment
    :param b1: first sine directional moment
    :param a2: second cosine directional moment
    :param b2: second sine directional moment
    :param direction: direction vector (in degrees)

    :return: estimated 2D directional spectrum
    """
    # convert degrees to radians
    direction_radians = np.deg2rad(direction)

    output_shape = list(a1.shape) + [len(direction)]
    if a1.ndim == 1:
        input_shape = [1, a1.shape[-1]]
    else:
        input_shape = [np.prod(a1.shape[0:-1]), a1.shape[-1]]

    a1 = a1.reshape(input_shape)
    b1 = b1.reshape(input_shape)
    a2 = a2.reshape(input_shape)
    b2 = b2.reshape(input_shape)
    D = root_finder(direction_radians, a1, b1, a2, b2)

    return D.reshape(output_shape) * (np.pi / 180) * e[..., None]