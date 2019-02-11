"""
Modelling of Non-Negative Linear Dynamical System (NN-LDS) under
beta-divergence cost constraints.

Cost optimization uses a majorization-minimization algorithm with
conditionally-weighted multiplicative updates.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/01/02
"""

import numpy as np

from tensortools.dynamics import LDS
from tensortools.operations import khatri_rao, unfold
from tensortools.tensors import KTensor

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
        LDS_dict={
            'axis': 0,
            'beta': 2,
            'lag_state': 1,
            'lag_exog': 1,
            'init': 'rand'
        },
        exog_input=None,
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

            lag_state: int, Lag-order corresponding to memory of state-transition.

            lag_exog: int, Lag-order corresponding to memory of control-input.

        exog_input: np.ndarray, shape: [t, p]
            If LDS_dict is used, then exog_input specifies the
            p-dimensional input signal, or control input, over time t.
            Must match the length of the observed axis.

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

        if (LDS_dict['lag_state'] < 0) or (LDS_dict['lag_exog'] < 0):
            raise Exception('LDS lag-order must be greater than 0.')

        if type(exog_input) != np.ndarray:
            raise Exception('LDS exogenous input must be a numpy array')

        if X.shape[LDS_dict['axis']] != exog_input.shape[0]:
            raise Exception(
                'Length of exogenous input does not match length of ' +
                'data tensor.')

    # Initialize model arrays/tensors.
    W, _ = optim_utils._get_initial_ktensor(NTF_dict['init'], X, rank,
                                            random_state)
    NTF_dict['W'] = W

    if LDS_dict is not None:
        A = optim_utils._get_initial_statematr(
            LDS_dict['init'], LDS_dict['lag_state'], rank, random_state)

        B = optim_utils._get_initial_controlmatr(
            LDS_dict['init'], LDS_dict['lag_exog'], rank, exog_input.shape[1],
            random_state)

        LDS_dict['AB'] = LDS(A, B)

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
        exog_input=None,
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
            Tensor containing dimensionality of the system output.
            Each Tensor fiber, I, is considered a mode of the system.
            Example modes are channels, time, trials, spectral frequency, etc.

        model : FitModel object
            Model that was created using the init_model function.

        exog_input: np.ndarray, shape: [t, p]
            If LDS_dict is used, then exogeneous input specifies the
            p-dimensional input signal, or control input, over time t.

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
    if exog_input is not None:
        if exog_input.shape[0] != X.shape[model.model_param['LDS']['axis']]:
            raise Exception(
                'Length of exogeneous input does not match length of ' +
                'data tensor.')

        if exog_input.shape[1] != model.model_param['LDS']['AB'].B.shape[-1]:
            raise Exception('Shape of input signal does not match shape of ' +
                            'control-input matrix.')

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
                    mp['LDS']['AB'].as_ord_1()

                    # Update H
                    WL = mp['LDS']['AB'].conv_state_to_lagged(W[n].T)
                    UL = mp['LDS']['AB'].conv_exog_to_lagged(exog_input.T)

                    lag_diff = mp['LDS']['AB'].lag_state - mp['LDS']['AB'].lag_exog
                    if lag_diff > 0:
                        UL = UL[:, int(np.abs(lag_diff)):]
                    elif lag_diff < 0:
                        WL = WL[:, int(np.abs(lag_diff)):]

                    neg1, pos1 = calc_time_grad(mp['LDS']['AB'].A, WL,
                                                mp['LDS']['AB'].B, UL,
                                                mp['LDS']['beta'])
                    neg1 = mp['LDS']['AB'].conv_state_to_unlagged(neg1)
                    pos1 = mp['LDS']['AB'].conv_state_to_unlagged(pos1)

                    neg += neg1.T
                    pos += pos1.T

                    mp['LDS']['AB'].as_ord_p()

            # vi) Update the observational component weights
            W[n] *= (neg / pos)**mm_gamma_func(mp['NTF']['beta'])
            if n == mp['LDS']['axis']:
                W[mp['LDS']['axis']] = (
                        W[mp['LDS']['axis']].T / np.sum(W[mp['LDS']['axis']], axis=1)).T


            # vii) Update the dynamical state weights
            if (flag_lds):
                if n == mp['LDS']['axis']:
                    mp['LDS']['AB'].as_ord_1()

                    # Update A/B
                    WL = mp['LDS']['AB'].conv_state_to_lagged(W[n].T)
                    UL = mp['LDS']['AB'].conv_exog_to_lagged(exog_input.T)

                    lag_diff = mp['LDS']['AB'].lag_state - mp['LDS']['AB'].lag_exog
                    if lag_diff > 0:
                        UL = UL[:, int(np.abs(lag_diff)):]
                    elif lag_diff < 0:
                        WL = WL[:, int(np.abs(lag_diff)):]

                    AX = mp['LDS']['AB'].A.dot(WL[:, :-1])
                    BU = mp['LDS']['AB'].B.dot(UL[:, :-1])

                    # Update A
                    neg, pos = calc_div_grad(WL[:, 1:], AX + BU, WL[:, :-1].T,
                                             mp['LDS']['beta'])

                    mp['LDS']['AB'].A *= \
                            (neg / pos)**mm_gamma_func(mp['LDS']['beta'])

                    # Update B
                    neg, pos = calc_div_grad(WL[:, 1:], AX + BU, UL[:, :-1].T,
                                             mp['LDS']['beta'])

                    mp['LDS']['AB'].B *= \
                            (neg / pos)**mm_gamma_func(mp['LDS']['beta'])

                    mp['LDS']['AB'].as_ord_p()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization model, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function

        # Cost of the observation model
        cost_obs = calc_cost(X, W.full(), mp['NTF']['beta'])

        # Update the model
        model.update(cost_obs)

    # end optimization loop, return model.
    return model.finalize()


def model_forecast(
        X,
        exog_input,
        model,
        fit_dict={
            'method': '{}-Divergence'.format(u'\u03B2'),
            'tol': 1e-5,
            'min_iter': 1,
            'max_iter': 500,
            'verbose': True
        }):
    """
    Use a trained NN-LDS model to forecast future states and observations.

    Parameters
    ----------
        X : np.ndarray, tensor_like with shape: [I_1, I_2, ..., I_N]
            Tensor containing dimensionality of the system output.
            Each Tensor fiber, I, is considered a mode of the system.
            Example modes are channels, time, trials, spectral frequency, etc.

        model : FitModel object
            Model that was created using the init_model function. The `model`
            must explicitly contain an LDS component.

        exog_input: np.ndarray, shape: [t, p]
            If LDS_dict is used, then exog_input specifies the
            p-dimensional input signal, or control input, over time t.
            Must match the length of the observed axis.

        forecast_steps: int
            Number of samples ahead to forecast using each time sample in X
            as a starting-point.

        fit_dict: dict, specifying fitting options.

            tol: float, Stopping tolerance for reconstruction error.

            max_iter: int, Max number of iterations to perform before exiting.

            min_iter: int, Min number of iterations to perform before exiting.

            verbose : bool, Display progress.

    Returns
    -------
    Xp : list[np.ndarray], listtensor_like with shape: [I_1, I_2, ..., I_N]
        Skeletal Tensor containing dimensionality of the system output.
            Each Tensor fiber, I, is considered a mode of the system.
            Example modes are channels, time, trials, spectral frequency, etc.
    """

    # Check model
    if 'NTF' not in model.model_param:
        raise Exception('Model does not have a observation component.')
    if 'LDS' not in model.model_param:
        raise Exception('Model does not have a dynamical system component.')

    # Check input matrix
    optim_utils._check_cpd_inputs(X, model.model_param['rank'])
    forecast_steps = (
        exog_input.shape[0] - X.shape[model.model_param['LDS']['axis']])
    if forecast_steps <= 0:
        raise Exception('Length of exogeneous input must be greater than ' +
                        'length of data tensor in order to forecast.')

    if exog_input.shape[1] != model.model_param['LDS']['AB'].B.shape[-1]:
        raise Exception('Shape of input signal does not match shape of ' +
                        'control-input matrix.')

    # Update model fit parameters
    model.set_fit_param(**fit_dict)

    # Reset the status of the model
    model.reset_status()

    # Set pointers to commonly used objects
    mp = model.model_param
    dAB = mp['LDS']['AB']
    ax_t = mp['LDS']['axis']

    # Initialize temporal state coefficients
    assert model.model_param['NTF']['init'] in ['rand', 'randn']
    if model.model_param['NTF']['init'] == 'randn':
        H = np.random.randn(X.shape[ax_t], model.model_param['rank'])
    else:
        H = np.random.rand(X.shape[ax_t], model.model_param['rank'])

    # Create a new model tensor with the temporal state mode replaced
    W = KTensor([mp['NTF']['W'][j] if j != ax_t else H for j in range(X.ndim)])
    mp['NTF']['W'] = W

    # Use observation model to estimate the current temporal state mode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while model.still_optimizing:

        # Select all components, but U_n
        components = [W[j] for j in range(X.ndim) if j != ax_t]

        # i)  Compute Khatri-Rao product
        kr = khatri_rao(components)
        Xn = unfold(X, ax_t)

        # ii) Compute unfolded prediction of X
        p = W[ax_t].dot(kr.T)

        # iii) Compute gradient for the observation model
        neg, pos = calc_div_grad(Xn, p, kr, mp['NTF']['beta'])

        # vi) Update the observational component weights
        W[ax_t] *= (neg / pos)**mm_gamma_func(mp['NTF']['beta'])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization model, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function

        # Cost of the observation model
        cost_obs = calc_cost(X, W.full(), mp['NTF']['beta'])

        # Update the model
        model.update(cost_obs)

    # end optimization loop.
    # Current temporal state mode is inferred, coefficients have been updated
    model.finalize()

    # Use LDS and current temporal state mode to forecast future state mode
    dAB.as_ord_p()

    Wn = list(W[ax_t])
    Un = list(exog_input)
    for p in range(forecast_steps):
        W_ix = range(len(Wn) - 1, len(Wn) - 1 - dAB.lag_state, -1)
        U_ix = range(len(Wn) - 1, len(Wn) - 1 - dAB.lag_exog, -1)

        AX = np.array([
            dAB.A[ii, :, :].dot(Wn[ij].reshape(-1, 1))
            for ii, ij in enumerate(W_ix)
            ])[:, :, 0].sum(axis=0)
        BU = np.array([
            dAB.B[ii, :, :].dot(Un[ij].reshape(-1, 1))
            for ii, ij in enumerate(U_ix)
            ])[:, :, 0].sum(axis=0)

        Wn.append(AX + BU)
    Wn = np.array(Wn)
    Wn = Wn[-forecast_steps:, :]

    # Re-mix forecasted state mode coefs through NTF
    XP = KTensor([W[j] if j != ax_t else Wn for j in range(X.ndim)])

    return XP
