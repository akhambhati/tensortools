"""
CP decomposition based on costs defined by beta-divergence family using
a majorization-minimization algorithm using conditionally-weighted
multiplicative updates.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/11/07
"""

import numpy as np
from tensortools.operations import khatri_rao, unfold
from tensortools.optimize import FitModel, optim_utils

from ._betadiv import calc_cost, calc_div_grad, calc_time_grad, mm_gamma_func
from ._var import (conv_A_to_var1, conv_A_to_varP, conv_X_to_var1,
                   conv_X_to_varP)


def ncp_betadiv(X,
                rank,
                observation_dict={'beta': 2},
                regularize_dict={'axis': None,
                                 'l1_ratio': 0.5,
                                 'alpha': 1e-6,
                                 'init': 'rand'},
                var_dict={'axis': None,
                          'beta': 2,
                          'lags': 1,
                          'init': 'rand'},
                random_state=None,
                fit_dict={
                    'tol': 1e-5,
                    'min_iter': 1,
                    'max_iter': 500,
                    'verbose': True
                }):
    """
    Fits nonnegative CP Decomposition using Beta-Divergence Cost Functions
    using Multiplicative Updates (MU) method.

    Parameters
    ----------
        X : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Tensor array (target) with nonnegative entries.

        rank : int
            The number of components to be computed.

        observation_dict: dict

            beta: float, beta-divergence parameter for the observation model.
                If 0: Itakura-Saito Divergence (scale-invariant)
                If 1: Kullback-Leibler Divergence
                If 2: Euclidean Distance
                Else: Parameterized version

        regularize_dict: dict

            axis: int, [0, N-1], axis of mode to apply Elastic-Net regularization.

            l1_ratio: float, [0, 1], rel. weight of l1 (sparsity) vs l2 (ridge).

            alpha: float, hyperparameter weight for Elastic-Net.

        var_dict: dict

            axis: int, [0, N-1], axis of mode corresponding to temporal dynamics.

            beta: float, beta-divergence parameter for the dynamical model.
                If 0: Itakura-Saito Divergence (scale-invariant)
                If 1: Kullback-Leibler Divergence
                If 2: Euclidean Distance
                Else: Parameterized version

            lags: int, VAR model order corresponding to `memory lag` behind time T.

        random_state: integer, RandomState instance or None
            If integer, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.

        init: str, or KTensor, optional (default ``'rand'``).
            Specifies initial guess for KTensor factor matrices.
            If ``'randn'``, Gaussian random numbers are used to initialize.
            If ``'rand'``, uniform random numbers are used to initialize.
            If KTensor instance, a copy is made to initialize the optimization.

        fit_dict: dict, specifying fitting options.

            tol: float, Stopping tolerance for reconstruction error.

            max_iter: int, Max number of iterations to perform before exiting.

            min_iter: int, Min number of iterations to perform before exiting.

            verbose : bool, Display progress.

    Returns
    -------
    model : FitModel instance
        Object which holds the fitted model. It provides the factor matrices
        in form of a KTensor, ``model.factors``.

    References
    ----------
    Févotte, Cédric, and Jérôme Idier. "Algorithms for nonnegative matrix
    factorization with the β-divergence."
    Neural computation 23.9 (2011): 2421-2456.
    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Check regularization params
    mode_reg = None
    if type(regularize_dict) == dict:
        mode_reg = regularize_dict[
            'axis'] if 'axis' in regularize_dict else False
        if mode_reg:
            mode_reg = mode_reg if ((mode_reg >= 0) &
                                    (mode_reg < X.ndim)) else False
        if mode_reg:
            assert ((regularize_dict['l1_ratio'] >= 0) &
                    (regularize_dict['l1_ratio'] <= 1))

    # Check VAR params
    mode_var = None
    if type(var_dict) == dict:
        mode_var = var_dict['axis'] if 'axis' in var_dict else False
        if mode_var:
            mode_var = mode_var if ((mode_var >= 0) &
                                    (mode_var < X.ndim)) else False
        if mode_var:
            assert var_dict['lags'] >= 1

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(observation_dict['init'], X, rank, random_state)
    A = optim_utils._get_initial_statematr(var_dict['init'], var_dict['lags'], rank, random_state) if mode_var is not None else None
    model = FitModel(factors=U,
                     states=A,
                     method='{}-Divergence'.format(u'\u03B2'),
                     **fit_dict)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while model.still_optimizing:

        for n in range(X.ndim):

            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            Xn = unfold(X, n)

            # ii) Compute unfolded prediction of X
            p = U[n].dot(kr.T)

            # iii) Compute gradient for the observation model
            neg, pos = calc_div_grad(Xn, p, kr, observation_dict['beta'])

            # iv) Add a regularizer
            if n == mode_reg:
                pos += (regularize_dict['alpha'] *
                        (2 * (1 - regularize_dict['l1_ratio']) * U[n] +
                         regularize_dict['l1_ratio']))

            # v) Compute gradient for the dynamical model
            if n == mode_var:
                AA = conv_A_to_var1(A)
                XX = conv_X_to_var1(U[n].T, lags=var_dict['lags'])

                neg1, pos1 = calc_time_grad(AA, XX, observation_dict['beta'])
                neg1 = conv_X_to_varP(neg1, lags=var_dict['lags'])
                pos1 = conv_X_to_varP(pos1, lags=var_dict['lags'])

                neg += neg1.T
                pos += pos1.T

                A = conv_A_to_varP(AA, lags=var_dict['lags'])

            # vi) Update the observational component weights
            U[n] *= (neg / pos)**mm_gamma_func(observation_dict['beta'])

            # vii) Update the dynamical state weights
            if n == mode_var:
                AA = conv_A_to_var1(A)
                XX = conv_X_to_var1(U[n].T, lags=var_dict['lags'])

                kr = khatri_rao([XX[:, :-1].T])
                Xn = XX[:, 1:]
                p = AA.dot(kr.T)
                neg, pos = calc_div_grad(Xn, p, kr, var_dict['beta'])

                AA *= (neg / pos)**mm_gamma_func(var_dict['beta'])
                A = conv_A_to_varP(AA, lags=var_dict['lags'])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization model, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function

        # Cost of the observation model
        cost_obs = calc_cost(X, U.full(), observation_dict['beta'])

        # Cost of the regularization component
        if mode_reg is not None:
            cost_reg = np.sum(
                regularize_dict['alpha'] *
                ((1 - regularize_dict['l1_ratio']
                  ) * np.linalg.norm(U[mode_reg], 2) +
                 regularize_dict['l1_ratio'] * np.linalg.norm(U[mode_reg], 1)))
        else:
            cost_reg = 0

        # Cost of the dynamical model
        if mode_var is not None:
            AA = conv_A_to_var1(A)
            XX = conv_X_to_var1(U[mode_var].T, lags=var_dict['lags'])
            cost_var = calc_cost(XX[:, 1:], AA.dot(XX[:, :-1]),
                                 var_dict['beta'])
        else:
            cost_var = 0

        # Update the model
        model.update(cost_obs + cost_reg + cost_var)

    # end optimization loop, return model.
    return model.finalize()
