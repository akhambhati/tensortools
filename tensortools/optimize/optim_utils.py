"""Utilities for summarizing and setting up optimization."""

import timeit

import numpy as np
import scipy as sci
from scipy import linalg
from tensortools.data.random_tensor import (rand_array, rand_ktensor,
                                            randn_array, randn_ktensor)
from tensortools.tensors import KTensor


def _check_cpd_inputs(X, rank):
    """Checks that inputs to optimization function are appropriate.

    Parameters
    ----------
    X : ndarray
        Tensor used for fitting CP decomposition.
    rank : int
        Rank of low rank decomposition.

    Raises
    ------
    ValueError: If inputs are not suited for CP decomposition.
    """
    if X.ndim < 2:
        raise ValueError("Array with X.ndim > 1 expected.")
    if rank <= 0 or not isinstance(rank, int):
        raise ValueError("Rank is invalid.")


def _get_initial_ktensor(init, X, rank, random_state, scale_norm=True):
    """
    Parameters
    ----------
    init : str
        Specifies type of initializations ('randn', 'rand')
    X : ndarray
        Tensor that the decomposition is fit to.
    rank : int
        Rank of decomposition
    random_state : RandomState or int
        Specifies seed for random number generator
    scale_norm : bool
        If True, norm is scaled to match X (default: True)

    Returns
    -------
    U : KTensor
        Initial factor matrices used optimization.
    normX : float
        Frobenious norm of tensor data.
    """

    normX = linalg.norm(X) if scale_norm else None

    if init == 'randn':
        # TODO - match the norm of the initialization to the norm of X.
        U = randn_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif init == 'rand':
        # TODO - match the norm of the initialization to the norm of X.
        U = rand_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif isinstance(init, KTensor):
        U = init.copy()

    else:
        raise ValueError("Expected 'init' to either be a KTensor or a string "
                         "specifying how to initialize optimization. Valid "
                         "strings are ('randn', 'rand').")

    return U, normX


def _get_initial_statematr(init, lag, rank, random_state, scale_norm=True):
    """
    Parameters
    ----------
    init : str
        Specifies type of initializations ('randn', 'rand')
    lag : int
        Memory inherent to the state transition matrix
    rank : int
        Number of states
    random_state : RandomState or int
        Specifies seed for random number generator
    scale_norm : bool
        If True, norm is scaled to match X (default: True)

    Returns
    -------
    A : np.ndarray, shape: [lag x rank x rank]
        Initial factor matrices used optimization.
    """

    if init == 'randn':
        A = randn_array((lag, rank, rank), random_state)

    elif init == 'rand':
        A = rand_array((lag, rank, rank), random_state)

    normA = np.linalg.norm(A) if scale_norm else 1
    A /= normA

    return A


class FitModel(object):
    """
    Holds result of optimization.

    Attributes
    ----------
    total_time: float
        Number of seconds spent before stopping optimization.
    obj : float
        Objective value of optimization (at current parameters).
    obj_hist : list of floats
        Objective values at each iteration.
    """

    def __init__(self, model_param={}):
        """Initializes FitModel.

        Parameters
        ----------
        model_param : dict
            Dictionary that holds all parameters pertaining to the model.
        """

        self.model_param = model_param
        self.fit_param = {}
        self.set_fit_param()

        self.status = {
            'obj': np.inf,
            'obj_hist': [],
            'iterations': 0,
            'converged': False,
            't0': timeit.default_timer(),
            'total_time': None
        }

    @property
    def still_optimizing(self):
        """True unless converged or maximum iterations/time exceeded."""

        # Check if we need to give up on optimizing.
        if ((self.status['iterations'] > self.fit_param['max_iter'])
                or (self.time_elapsed() > self.fit_param['max_time'])):
            return False

        # Always optimize for at least 'min_iter' iterations.
        elif (('improvement' not in self.status)
              or (self.status['iterations'] < self.fit_param['min_iter'])):
            return True

        # Check convergence.
        else:
            self.status['converged'] = \
                    self.status['improvement'] < self.fit_param['tol']
            return False if self.status['converged'] else True

    def time_elapsed(self):
        return timeit.default_timer() - self.status['t0']

    def update(self, obj):

        # Keep track of iterations.
        self.status['iterations'] += 1

        # Compute improvement in objective.
        self.status['improvement'] = self.status['obj'] - obj
        self.status['obj'] = obj
        self.status['obj_hist'].append(obj)

        # If desired, print progress.
        if self.fit_param['verbose']:
            p_args = (self.fit_param['method'], self.status['iterations'],
                      self.status['obj'], self.status['improvement'])
            s = '{}: iteration {}, objective {}, improvement {}.'
            print(s.format(*p_args))

    def finalize(self):

        # Set final time, final print statement
        self.status['total_time'] = self.time_elapsed()

        if self.fit_param['verbose']:
            s = 'Converged after {} iterations, {} seconds. Objective: {}.'
            print(
                s.format(self.status['iterations'], self.status['total_time'],
                         self.status['obj']))

        return self

    def set_fit_param(self,
                      method=None,
                      tol=1e-5,
                      verbose=True,
                      max_iter=500,
                      min_iter=1,
                      max_time=np.inf):
        """Set the parameters of the model fitting.

        Parameters
        ----------
        method : str, Name of optimization method (used for printing).

        tol : float, Stopping criterion.

        verbose : bool, Whether to print progress of optimization.

        max_iter : int, Max iterations before quitting early.

        min_iter : int, Min iterations before stopping due to convergence.

        max_time : float, Max seconds before quitting early.
        """

        self.fit_param['method'] = method
        self.fit_param['tol'] = tol
        self.fit_param['verbose'] = verbose
        self.fit_param['max_iter'] = max_iter
        self.fit_param['min_iter'] = min_iter
        self.fit_param['max_time'] = max_time
