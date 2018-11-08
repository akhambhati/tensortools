"""
Functions to manipulate arrays for Vector Autoregressive modelling.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/11/07
"""

import numpy as np


def conv_A_to_var1(A):
    """Convert a p-lagged state-transition matrix to 1-lagged matrix."""

    assert A.ndim == 3
    l, k, _ = A.shape

    AA = np.eye(k * l, k=-k)
    AA[:k, :] = A.transpose((1, 0, 2)).reshape(k, -1)

    return AA


def conv_A_to_varP(A, lags):
    """Convert a 1-lagged state-transition matrix to p-lagged matrix."""

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    kl, _ = A.shape
    k = kl / lags
    assert k == np.round(k)
    k = int(np.round(k))

    return A[:k, :].reshape(k, -1, k).transpose(1, 0, 2)


def conv_X_to_var1(X, lags):
    """Convert unlagged signal matrix into a stack, lagged, signal matrix"""
    # X has shape: [K, N]

    assert X.ndim == 2
    K, N = X.shape
    assert N > lags

    # Shift X to shape (L, K, N-L+1)
    # Return an unfolded array (K*L, N-L+1)
    X = np.array(
        [X[:, l:(N - (lags - l) + 1)] for l in range(lags - 1, -1, -1)])

    return X.reshape(-1, N - lags + 1)


def conv_X_to_varP(X, lags):
    """Convert stack, lagged signal matrix into an unlagged signal matrix"""

    assert X.ndim == 2
    KL, N = X.shape
    K = KL / lags
    assert K == np.round(K)
    K = int(np.round(K))

    X = X.reshape(lags, K, N)

    # Reconstruct the shortened signal
    # New X has shape: [K, N+lags-1]
    X = np.concatenate((X[-1, :, :], X[0, :, 1 - lags:]), axis=1)

    return X
