"""
Modelling of Non-Negative Linear Dynamical System (NN-LDS) under
beta-divergence cost constraints.

Cost optimization uses a majorization-minimization algorithm with
conditionally-weighted multiplicative updates.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/11/14
"""

import numpy as np
from tensortools.dynamics import LDS
from tensortools.operations import khatri_rao, unfold

from . import optim_utils
from ._betadiv import calc_cost, calc_div_grad, calc_time_grad, mm_gamma_func


def init_model(
        X,
        rank,
        NTF_dict={'beta': 2,
                  'init': 'rand'},
        REG_dict={'axis': 0,
                  'l1_ratio': 0.5,
                  'alpha': 1e-6,
                  'init': 'rand'},
        LDS_dict={'axis': 0,
                  'beta': 2,
                  'lags': 1,
                  'init': 'rand'},
        random_state=None):
    """
    Initialize a FitModel object with parameters for NN-LDS.

    Parameters
    ----------
        X : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Skeletal Tensor containing dimensionality of the system output.
            Each Tensor fiber, I, is considered a mode of the system. 
            Example modes are channels, time, trials, spectral frequency, etc.

        rank : int
            Low-dimensional sub-space of the system.

        NTF_dict: dict
            Parameters corresponding to the observation model (tensor fac).

            beta: float, beta-divergence parameter for the observation model.
                If 0: Itakura-Saito Divergence (scale-invariant)
                If 1: Kullback-Leibler Divergence
                If 2: Euclidean Distance
                Else: Parameterized version

            init: str, ['rand', 'randn']
                Specifies initial guess for KTensor factor matrices.
                If ``'randn'``, Gaussian random numbers are used to initialize.
                If ``'rand'``, uniform random numbers are used to initialize.

        REG_dict: dict
            Parameters corresponding to model regularization via Elastic-Net.

            axis: int, [0, N-1], axis to apply Elastic-Net regularization.
                If set to None, no regularization will be applied.

            l1_ratio: float, [0, 1], rel. wt. of l1 (sparsity) vs l2 (ridge).

            alpha: float, hyperparameter weight for Elastic-Net.

        LDS_dict: dict
            Parameters corresponding to the dynamical model.
            If set to None, no LDS modelling will be applied.

            axis: int, [0, N-1], axis corresponding to temporal dynamics.

            beta: float, beta-divergence parameter for the dynamical model.
                If 0: Itakura-Saito Divergence (scale-invariant)
                If 1: Kullback-Leibler Divergence
                If 2: Euclidean Distance
                Else: Parameterized version

            lags: int, Lag-order corresponding to system memory.

        random_state: integer, RandomState instance or None
            If integer, specifies seed used by the random number generator;
            If RandomState, specifies object of the random number generator;
            If None, np.random spawns its own RandomState instance.

    Returns
    -------
    model : FitModel instance
        Object which holds the fitted model. It provides the factor matrices
        in form of a KTensor, ``model.factors``.
    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)
    n_mode = X.ndim
    if NTF_dict is None:
        raise Exception('Parameters for observation model must be specified.')

    # Check regularization params
    if type(REG_dict) == dict:
        if ((REG_dict['axis'] < 0) or (REG_dict['axis'] >= n_mode)
                or (type(REG_dict['axis']) != int)):
            raise Exception('Regularization specified to non-existing mode.')

        if ((REG_dict['l1_ratio'] < 0) or (REG_dict['l1_ratio'] > 1)):
            raise Exception('Regularization l1-ratio must be between 0 and 1.')

    # Check LDS params
    if type(LDS_dict) == dict:
        if ((LDS_dict['axis'] < 0) or (LDS_dict['axis'] >= n_mode)
                or (type(LDS_dict['axis']) != int)):
            raise Exception('LDS constraint specified to non-existing mode.')

        if (LDS_dict['lags'] < 0):
            raise Exception('LDS lag-order must be greater than 0.')

    # Initialize model arrays/tensors.
    W, _ = optim_utils._get_initial_ktensor(NTF_dict['init'], X, rank,
                                            random_state)
    NTF_dict['W'] = W

    if LDS_dict is not None:
        A = LDS(
            optim_utils._get_initial_statematr(
                LDS_dict['init'], LDS_dict['lags'], rank, random_state))
        LDS_dict['A'] = A

    model = optim_utils.FitModel(model_param={
        'rank': rank,
        'NTF': NTF_dict,
        'LDS': LDS_dict,
        'REG': REG_dict
    })

    return model


def model_update(
        X,
        model,
        fixed_axes=[],
        fit_dict={
            'method': '{}-Divergence'.format(u'\u03B2'),
            'tol': 1e-5,
            'min_iter': 1,
            'max_iter': 500,
            'verbose': True
        }):
    """
    Update the model parameters by optimizing Beta-Divergence Cost Functions
    using Multiplicative Updates (MU) method.

    Parameters
    ----------
        X : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Skeletal Tensor containing dimensionality of the system output.
            Each Tensor fiber, I, is considered a mode of the system.
            Example modes are channels, time, trials, spectral frequency, etc.

        model : FitModel object
            Model that was created using the init_model function.

        fixed_axes: None, int, or list[int]
            Modes of the model to keep constant during the update.
            Typically used to test the model on new data by keeping
            "basis modes" fixed and updating "activation" coefficients.

            If list[int], fix modes corresponding to axes in X for each int.
                An empty list implies that all modes get updated.

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

    # Check input matrix
    optim_utils._check_cpd_inputs(X, model.model_param['rank'])
    if X.shape != model.model_param['NTF']['W'].shape:
        raise Exception('Shape of input X does not match shape expected by ' +
                        'initialized model.')

    # Check fixed axes
    if type(fixed_axes) is not list:
        raise Exception('Fixed axes must be list of axis indices')
    if not all([
            True if (int(a) == a) and (a >= 0) and (a < X.ndim) else False
            for a in fixed_axes
    ]):
        raise Exception('Fixed axes not integers or exceed dimensions of X.')
    fixed_axes = [int(a) for a in fixed_axes]

    # Update model fit parameters
    model.set_fit_param(**fit_dict)

    # Reset the status of the model
    model.reset_status()

    # Set pointers to commonly used objects
    mp = model.model_param
    W = mp['NTF']['W']

    # Set flags for conditional operations
    flag_lds = True if mp['LDS'] is not None else False
    flag_reg = True if mp['REG'] is not None else False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while model.still_optimizing:

        for n in range(X.ndim):

            # If n corresponds to one of the fixed axes then don't update
            if n in fixed_axes:
                continue

            # Select all components, but U_n
            components = [W[j] for j in range(X.ndim) if j != n]

            # i)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            Xn = unfold(X, n)

            # ii) Compute unfolded prediction of X
            p = W[n].dot(kr.T)

            # iii) Compute gradient for the observation model
            neg, pos = calc_div_grad(Xn, p, kr, mp['NTF']['beta'])

            # iv) Add a regularizer
            if (flag_reg):
                if n == mp['REG']['axis']:
                    pos += (mp['REG']['alpha'] *
                            (2 * (1 - mp['REG']['l1_ratio']) * W[n] +
                             mp['REG']['l1_ratio']))

            # v) Compute gradient for the dynamical model
            if (flag_lds):
                if n == mp['LDS']['axis']:
                    mp['LDS']['A'].as_ord_1()
                    WL = mp['LDS']['A'].conv_X_to_lagged(W[n].T)

                    neg1, pos1 = calc_time_grad(mp['LDS']['A'].A, WL,
                                                mp['LDS']['beta'])
                    neg1 = mp['LDS']['A'].conv_X_to_unlagged(neg1)
                    pos1 = mp['LDS']['A'].conv_X_to_unlagged(pos1)

                    neg += neg1.T
                    pos += pos1.T

                    mp['LDS']['A'].as_ord_p()

            # vi) Update the observational component weights
            W[n] *= (neg / pos)**mm_gamma_func(mp['NTF']['beta'])

            # vii) Update the dynamical state weights
            if (flag_lds):
                if n == mp['LDS']['axis']:
                    mp['LDS']['A'].as_ord_1()
                    WL = mp['LDS']['A'].conv_X_to_lagged(W[n].T)

                    kr = khatri_rao([WL[:, :-1].T])
                    Xn = WL[:, 1:]
                    p = mp['LDS']['A'].A.dot(kr.T)
                    neg, pos = calc_div_grad(Xn, p, kr, mp['LDS']['beta'])

                    mp['LDS']['A'].A *= \
                            (neg / pos)**mm_gamma_func(mp['LDS']['beta'])

                    mp['LDS']['A'].as_ord_p()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization model, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function

        # Cost of the observation model
        cost_obs = calc_cost(X, W.full(), mp['NTF']['beta'])

        # Cost of the regularization component
        if (flag_reg):
            l1 = mp['REG']['l1_ratio'] * \
                    np.linalg.norm(W[mp['REG']['axis']], 1)
            l2 = (1 - mp['REG']['l1_ratio']) * \
                    np.linalg.norm(W[mp['REG']['axis']], 2)
            cost_reg = np.sum(mp['REG']['alpha'] * (l1 + l2))
        else:
            cost_reg = 0

        # Cost of the dynamical model
        if (flag_lds):
            mp['LDS']['A'].as_ord_1()
            WL = mp['LDS']['A'].conv_X_to_lagged(W[mp['LDS']['axis']].T)
            cost_var = calc_cost(WL[:, 1:], mp['LDS']['A'].A.dot(WL[:, :-1]),
                                 mp['LDS']['beta'])
            mp['LDS']['A'].as_ord_p()
        else:
            cost_var = 0

        # Update the model
        model.update(cost_obs + cost_reg + cost_var)

    # end optimization loop, return model.
    return model.finalize()
