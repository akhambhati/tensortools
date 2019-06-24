"""
The beta-divergence represents a family of cost functions that lay along a
continuum of three well-known cost functions that are defined by specific
values of beta:
    - beta = 0 : Itakura-Saito divergence (underlying mult. Gamma noise)
    - beta = 1 : Kullback-Leibler divergence (underlying mult. Poisson noise)
    - beta = 2 : Euclidean distance (underlying add. Gaussian noise)

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/11/06
"""

import numpy as np

def calc_cost(x, x_h, beta):
    """
    Compute the beta divergence between two matrices for a given beta.

    Parameters
    ----------
        x : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Tensor array (target) with nonnegative entries.

        x_h : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Tensor array (approximation) with nonnegative entries.

        beta : float
            Parameter that defines the specific beta-divergence cost function.

    Return
    ------
        cost: float
            The divergence or loss between x and y.
    """

    if x.shape != x_h.shape:
        raise Exception('x and x_h must have same array shape')

    if beta == 2:
        return np.sqrt(np.sum((x - x_h)**2))

    elif beta == 1:
        return np.sum(x * np.log(x / x_h) - x + x_h)

    elif beta == 0:
        return np.sum((x / x_h) - np.log(x / x_h) - 1)

    else:
        return np.sum(1 / (beta * (beta - 1)) *
                      (x**beta + (beta - 1) * x_h**beta - beta * x * x_h**
                       (beta - 1)))


def mm_gamma_func(beta):
    """Define the gamma function for majorization-minimzation algorithm"""

    if (beta < 1):
        return 1 / (2 - beta)

    if (beta >= 1) & (beta <= 2):
        return 1

    if (beta > 2):
        return 1 / (beta - 1)


def calc_div_grad(x, x_h, kr, beta):
    """
    Compute the positive and negative gradient components for the
    beta divergence.

    Parameters
    ----------
        x : np.ndarray, unfolded tensor with shape [I_N, prod(diff(I_*, I_N))]
            Tensor array (target) with nonnegative entries.

        x_h : np.ndarray, unfolded tensor with shape [I_N, prod(diff(I_*, I_N))]
            Tensor array (approximate) with nonnegative entries.

        kr : np.ndarray, Khatri-Rao tensor with shape [prod(diff(I_*, I_N)), K]
            Tensor array for the Khatri-Rao product of all held variables in the
            partial derivative.

        beta : float
            Parameter that defines the specific cost function.

    Return
    ------
        neg: array_like
            Negative gradient component

        pos: array_like
            Positive gradient component
    """

    neg_inv = x_h**(beta - 2)
    neg_inv[~np.isfinite(neg_inv)] = 0
    neg = (neg_inv * x).dot(kr)

    pos_inv = x_h**(beta - 1)
    pos_inv[~np.isfinite(pos_inv)] = 0
    pos = (pos_inv).dot(kr)

    return neg, pos


def calc_time_grad(A, X_t, B, U_t, beta):
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
            Represents X_t_minus_1.

        B : np.ndarray shape: [k, p]
            Control-input matrix (can be of arbitrary lags).

        U_t : np.ndarray shape: [p, N]
            Stacked time exogenous input vector (can be of arbitrary lags).
            Represents U_t_minus_1.

        beta : float
            Parameter that defines the specific cost function.

    Return
    ------
        neg: array_like
            Negative gradient component

        pos: array_like
            Positive gradient component
    """

    K, N = X_t.shape

    # Setup placeholders
    neg_forw = np.zeros_like(X_t)
    pos_forw = np.zeros_like(X_t)
    neg_back = np.zeros_like(X_t)
    pos_back = np.zeros_like(X_t)

    X_t0 = X_t[:, :-2]
    X_t1 = X_t[:, 1:-1]
    X_t2 = X_t[:, 2:]

    U_t0 = U_t[:, :-2]
    U_t1 = U_t[:, 1:-1]
    U_t2 = U_t[:, 2:]


    # Compute the forward gradients (t --> t+1)
    AXBU = A.dot(X_t0) + B.dot(U_t0)

    neg_inv = AXBU**(beta - 2)
    neg_inv[~np.isfinite(neg_inv)] = 0
    neg_forw[:, :-2] = A.T.dot(neg_inv) * X_t1

    pos_inv = AXBU**(beta - 1)
    pos_inv[~np.isfinite(pos_inv)] = 0
    pos_forw[:, :-2] = A.T.dot(pos_inv)

    # Compute the reverse gradients (t-1 --> t)
    AXBU = A.dot(X_t1) + B.dot(U_t1)
    if beta > 1:
        neg_inv = AXBU**(beta - 1)
        neg_inv[~np.isfinite(neg_inv)] = 0
        neg_back[:, 2:] = np.abs(1 / (beta - 1)) * (neg_inv)

        pos_inv = X_t2**(beta - 1)
        pos_inv[~np.isfinite(pos_inv)] = 0
        pos_back[:, 2:] = np.abs(1 / (beta - 1)) * (pos_inv)

    if beta < 1:
        neg_inv = X_t2**(beta - 1)
        neg_inv[~np.isfinite(neg_inv)] = 0
        neg_back[:, 2:] = np.abs(1 / (beta - 1)) * (neg_inv)

        pos_inv = AXBU**(beta - 1)
        pos_inv[~np.isfinite(pos_inv)] = 0
        pos_back[:, 2:] = np.abs(1 / (beta - 1)) * (pos_inv)

    if beta == 1:
        neg_back[:, 2:] = np.log(AXBU)
        pos_back[:, 2:] = np.log(X_t2)

    return (neg_back + neg_forw), (pos_back + pos_forw)
