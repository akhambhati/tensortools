"""
The beta-divergence represents a family of cost functions that lay along a
continuum of three well-known cost functions that are defined by specific
values of beta:
    - beta = 0 : Itakura-Saito divergence (underlying mult. Gamma noise)
    - beta = 1 : Kullback-Leibler divergence (underlying mult. Poisson noise)
    - beta = 2 : Euclidean distance (underlying add. Gaussian noise)

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/11/03
"""

import numpy as np


def calc_cost(x, x_h, beta):
    """
    Compute the beta divergence between two matrices for a given beta.

    Parameters
    ----------
        x : (I_1, ..., I_N) array_like
            A real array (target) with nonnegative entries.

        x_h : (I_1, ..., I_N) array_like
            A real array (approximation) with nonnegative entries.

        beta : float
            Parameter that defines the specific cost function.

    Return
    ------
        cost: float
            The divergence or loss between x and y.
    """

    if x.shape != x_h.shape:
        raise Exception('x and x_h must have same array shape')

    if beta == 2:
        return np.linalg.norm(x - x_h)

    elif beta == 1:
        return np.sum(x * np.log(x / x_h) - x + x_h)

    elif beta == 0:
        return np.sum((x / x_h) - np.log(x / x_h) - 1)

    else:
        return np.sum(1 / (beta * (beta - 1)) *
                      (x**beta + (beta - 1) * x_h**beta - beta * x * x_h**
                       (beta - 1)))


def _mm_gamma_func(beta):
    """Define the gamma function for MM-algorithm based on beta value"""

    if (beta < 1):
        return 1 / (2 - beta)

    if (beta >= 1) & (beta <= 2):
        return 1

    if (beta > 2):
        return 1 / (beta - 1)


def calc_div_grad(x, x_h, kr, beta, alg='heuristic'):
    """
    Compute the positive and negative gradient components for the
    beta divergence.

    Parameters
    ----------
        x : (I_1, ..., I_N) array_like
            A real array (target) with nonnegative entries.

        x_h : (I_1, ..., I_N) array_like
            A real array (approximation) with nonnegative entries.

        kr : (I_1, ..., I_N) array_like
            A real array for the Khatri-Rao product of all held variables in the
            partial derivative.

        beta : float
            Parameter that defines the specific cost function.

        alg: ['maxmin', 'heuristic']
            The algorithm used to weight the gradients depending on desired
            update speed.
            - 'maxmin' - maximization-minimization algorithm
            - 'heuristic' - heuristic algorithm

    Return
    ------
        neg: array_like
            Negative gradient component
        pos: array_like
            Positive gradient component
    """

    alg = alg.lower()
    if alg == 'maxmin':
        pow_fac = _mm_gamma_func(beta)
    elif alg == 'heuristic':
        pow_fac = 1
    else:
        raise Exception('Specified algorithm not supported.')

    neg = (x_h**(beta - 2) * x).dot(kr)
    pos = (x_h**(beta - 1)).dot(kr)

    return neg, pos


def calc_time_grad(A, X_t, beta, alg='heuristic'):
    """
    Compute the positive and negative gradient components for updating
    an X_t. Used as a constraint condition in conjunction with the more general
    divergence gradient.

    X_t = A * X_t_minus_1

    Parameters
    ----------
        A : np.ndarray shape: [k, k]
            State-transition matrix (can be of arbitrary lags).

        X_t : np.ndarray shape: [k, N]
            Stacked time observation matrix (can be of arbitrary lags).

        beta : float
            Parameter that defines the specific cost function.

        alg: ['maxmin', 'heuristic']
            The algorithm used to weight the gradients depending on desired
            update speed.
            - 'maxmin' - maximization-minimization algorithm
            - 'heuristic' - heuristic algorithm

    Return
    ------
        neg: array_like
            Negative gradient component
        pos: array_like
            Positive gradient component
    """

    alg = alg.lower()
    if alg == 'maxmin':
        pow_fac = _mm_gamma_func(beta)
    elif alg == 'heuristic':
        pow_fac = 1
    else:
        raise Exception('Specified algorithm not supported.')

    K, N = X_t.shape

    # Setup placeholders
    neg_forw = np.zeros_like(X_t)
    pos_forw = np.zeros_like(X_t)

    X_t0 = X_t[:, :-1]
    X_t1 = X_t[:, 1:]

    # Compute the forward gradients (t --> t+1)
    neg_forw[:, :-1] = A.T.dot(A.dot(X_t0)**(beta - 2)) * X_t1
    pos_forw[:, :-1] = A.T.dot(A.dot(X_t0)**(beta - 1))

    return neg_forw, pos_forw
