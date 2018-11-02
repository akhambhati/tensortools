"""
CP decomposition based on costs defined by beta-divergence family.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
"""

import numpy as np
import scipy as sci
from scipy import linalg

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils

from .._hals_update import _hals_update

def _beta_div(X, U, beta):
    """Define the different divergences"""

    if beta == 2:
        return np.linalg.norm(X - U)

    elif beta == 1:
        return np.sum(
            X * np.log(X/U) - X + U
        )

    elif beta == 0:
        return np.sum(
            X/U - np.log(X/U) - 1
        )

    else:
        return np.sum(1/(beta*(beta-1)) * 
                (X**beta + (beta-1)*U**beta - beta*X*U**(beta-1))
        )

def ncp_betadiv(X, rank, beta, random_state=None, init='rand', **options):
    """
    Fits nonnegative CP Decomposition using Beta-Divergence Cost Functions
    using Multiplicative Updates (MU) method.

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with nonnegative entries and ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    beta : float
        Specify the beta-divergence parameter.
        If 0: Itakura-Saito Divergence (scale-invariant)
        If 1: Kullback-Leibler Divergence
        If 2: Euclidean Distance
        Else: Parameterized version

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    init : str, or KTensor, optional (default ``'rand'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    This implemenation is using the Hierarcial Alternating Least Squares Method.


    References
    ----------
    Févotte, Cédric, and Jérôme Idier. "Algorithms for nonnegative matrix
    factorization with the β-divergence."
    Neural computation 23.9 (2011): 2421-2456.
    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, '{}-Divergence'.format(u'\u03B2'), **options)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while result.still_optimizing:

        violation = 0.0

        for n in range(X.ndim):

            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            Xn = unfold(X, n)

            # ii) Update component U_n
            if beta == 2:
                grams = sci.multiply.reduce([arr.T.dot(arr) for arr in components])
                p = Xn.dot(kr)

                violation += _hals_update(U[n], grams, p)

            else:
                p = U[n].dot(kr.T)

                num = (p**(beta-2) * Xn).dot(kr)
                den = (p**(beta-1)).dot(kr)
                U[n] *= num / den

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        result.update(_beta_div(X, U.full(), beta))

    # end optimization loop, return result.
    return result.finalize()
