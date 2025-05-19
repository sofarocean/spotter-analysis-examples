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
from scipy.optimize import minimize

# Entry Function
# =============================================================================
def mem_distribution(theta, lambdas):
    N = len(lambdas) // 2
    lambda_cos = lambdas[:N]
    lambda_sin = lambdas[N:]
    exponent = sum(
        lambda_cos[n] * np.cos((n + 1) * theta) +
        lambda_sin[n] * np.sin((n + 1) * theta)
        for n in range(N)
    )
    unnormalized = np.exp(exponent)
    return unnormalized / np.trapz(unnormalized, theta)

# Objective function for optimization
def mem_objective(lambdas, theta, target_moments):
    N = len(target_moments) // 2
    D = mem_distribution(theta, lambdas)
    moments = []
    for n in range(1, N + 1):
        moments.append(np.trapz(D * np.cos(n * theta), theta))  # a_n
        moments.append(np.trapz(D * np.sin(n * theta), theta))  # b_n
    return np.sum((np.array(moments) - np.array(target_moments))**2)


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
    direction_radians = np.deg2rad(direction)

    ndirs = len(direction)
    nfreqs = e.shape[-1]

    # Preallocate result: shape (nfreqs, n_dirs)
    D = np.zeros((nfreqs, ndirs))

    n_order = 2
    a_moments = np.array([a1, a2]).T
    b_moments = np.array([b1, b2]).T
    # Loop over each frequency
    for i in range(nfreqs):
        target_moments = []
        for n in range(n_order):
            target_moments.append(a_moments[i, n])
            target_moments.append(b_moments[i, n])

        init_guess = np.zeros(len(target_moments))  # initial guess for optimization
        res = minimize(mem_objective, init_guess, args=(direction_radians, target_moments), method='BFGS')

        D[i, :] = mem_distribution(direction_radians, res.x)

    return D * (np.pi / 180) * e[..., None]