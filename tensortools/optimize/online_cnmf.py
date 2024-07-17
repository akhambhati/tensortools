"""
Modelling of Non-Negative Linear Dynamical System (NN-LDS) under
beta-divergence cost constraints.

Cost optimization uses a majorization-minimization algorithm with
conditionally-weighted multiplicative updates.

Author: Ankit N. Khambhati <akhambhati@gmail.com>
Last Updated: 2018/01/02
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import tensorly as tl

from tensortools.dynamics import LDS
from tensortools.tensors import KTensor

from . import optim_utils
from ._betadiv import calc_cost, calc_div_grad, calc_time_grad, mm_gamma_func


EPSILON = np.finfo(np.float64).eps


def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def model_prediction(W):
    P = [shift(W[0], i, 0) @ W[i+1].T for i in range(len(W.shape)-1)]
    P = np.array(P).sum(axis=0)

    return P


def init_model(
        n_feat,
        conv_size,
        rank,
        beta=2,
        MB_dict={'update_rate': 1,
                 'update_scaling': 1,
                 'update_burn': 0},
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

            init: numpy.random.[XYZ]
                Specifies initial guess for factor matrices.
                Draw values from the specified random distribution

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
    dict_dim = (conv_size, ) + (n_feat,) * conv_size
    n_mode = len(dict_dim)

    # Initialize model arrays/tensors.
    NTF_dict = {'beta': beta}
    W = []
    for m_i in range(n_mode):
        W.append(tl.tensor(np.random.rand(dict_dim[m_i], rank)))
        if m_i > 0:
            norm = np.array([np.linalg.norm(w) for w in W[m_i].T])
            W[m_i] = W[m_i] / norm

    NTF_dict['W'] = KTensor(W)

    NEG = []
    POS = []
    for m_i in range(n_mode):
        NEG.append(np.zeros((dict_dim[m_i], rank)))
        POS.append(np.zeros((dict_dim[m_i], rank)))
    NTF_dict['NEG_Update'] = NEG
    NTF_dict['POS_Update'] = POS

    TRAIN_dict = {
            'minibatch': MB_dict,
            'total_epochs': 0,
            'epochs_since_update': 0}

    model = optim_utils.FitModel(model_param={
        'n_mode': n_mode,
        'n_feat': n_feat,
        'conv_size': conv_size,
        'rank': rank,
        'NTF': NTF_dict,
        'TRAIN': TRAIN_dict,
    })

    return model


def model_filter(
        X,
        model,
        reset=True,
        fit_dict={
            'method': '{}-Divergence'.format(u'\u03B2'),
            'tol': 1e-8,
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

    # Update model fit parameters
    model.set_fit_param(**fit_dict)

    # Set pointers to commonly used objects
    mp = model.model_param
    W = mp['NTF']['W']

    pred = model_prediction(W)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    n = 0

    # Reset the status of the model
    if reset:
        model.reset_status()
        W[n][...] = np.random.rand(W[n].shape[0], W[n].shape[1])

    while model.still_optimizing:

        neg = 0
        pos = 0
        for l in range(mp['n_mode']):

            # If n corresponds to one of the fixed axes then don't update
            if l == 0:
                continue

            # Select all components, but U_n
            # i)  Compute Khatri-Rao product
            kr = W.factors[l]

            # iii) Compute gradient for the observation model
            neg_, pos_ = calc_div_grad(shift(X, 1-l, 0), pred, kr, mp['NTF']['beta'])
            neg += neg_
            pos += pos_

        # iv) Update the observational component weights
        neg_pos_grad = (neg / pos)**mm_gamma_func(mp['NTF']['beta'])
        W[n][...] *= neg_pos_grad

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization model, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function

        # Cost of the observation model
        cost_obs = calc_cost(X, model_prediction(W), mp['NTF']['beta'])

        # Update the model
        model.update(cost_obs)

    # end optimization loop, return model.
    return W[n]


def model_update_W(
        X,
        model):
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

    # Set pointers to commonly used objects
    mp = model.model_param
    W = mp['NTF']['W']

    pred = model_prediction(W).T

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fixed_mode = [0]

    for n in range(mp['n_mode']):

        # If n corresponds to one of the fixed axes then don't update
        if n in fixed_mode:
            continue

        # Select all components, but U_n

        # i)  Compute Khatri-Rao product
        kr = shift(W[0], n-1, 0)

        # iii) Compute gradient for the observation model
        neg, pos = calc_div_grad(X.T, pred, kr, mp['NTF']['beta'])

        # vi) Update the observational component weights
        mp['NTF']['NEG_Update'][n] += (neg * (W[n]**(1/mm_gamma_func(mp['NTF']['beta']))))
        mp['NTF']['POS_Update'][n] += pos

    mp['TRAIN']['total_epochs'] += 1
    mp['TRAIN']['epochs_since_update'] += 1

    cost = np.nan*np.zeros(mp['rank'])
    if ((mp['TRAIN']['epochs_since_update'] >=
         mp['TRAIN']['minibatch']['update_rate']) &
        (mp['TRAIN']['total_epochs'] >
         mp['TRAIN']['minibatch']['update_burn'])):

        W_old = W.copy()
        error = []
        for n in range(mp['n_mode']):

            # If n corresponds to one of the fixed axes then don't update
            if n in fixed_mode:
                continue

            W[n] = (mp['NTF']['NEG_Update'][n] /
                    mp['NTF']['POS_Update'][n])**mm_gamma_func(mp['NTF']['beta'])

            norm = np.array([np.linalg.norm(w) for w in W[n].T])

            W[n] /= norm
            mp['NTF']['NEG_Update'][n] /= norm
            mp['NTF']['POS_Update'][n] *= norm

        error.append(np.sum((W_old[n] - W[n])**2, axis=0))
        cost = np.array(error).sum(axis=0)
        cost = np.sqrt(cost)

        mp['epochs_since_update'] = 0

    return cost
