"""
Class to instantiate a Linear Dynamical System model

Author: Ankit N. Khambhati
Last Updated: 2018/12/29
"""
import numpy as np


class LDS(object):
    """Linear Dynamical System object.

    Attributes
    ----------
    A : np.ndarray; shape: [lag_state x rank x rank]
        State matrix containing transition coefficients.
    B : np.ndarray; shape: [lag_exog x rank x exog]
        Control-input matrix mapping exogenous input to state transitions.
    rank_state : int
        Rank of the state-space of the system.
    rank_exog : int
        Rank of the input-space of the system.
    lag_state : int
        Number of time-lags represented by the state-transition component.
    lag_exog : int
        Number of time-lags represented by the exogenous input component.
    form : ['ord_1', 'ord_p']
        Specifies the current formulation of the state and control matrices.
        A p-th order model can always be represented as a 1-st order model.
    """

    def __init__(self, A, B):
        """Initializes LDS.

        Parameters
        ----------
        A : np.ndarray; shape: [n_lag x n_rank x n_rank]
            State matrix (in ord_p form) containing transition coefficients.
        B : np.ndarray; shape: [n_lag x n_rank x n_exog]
            Control-input matrix (in ord_p form) containing 
            state-control coefficients.
        """

        if A.ndim not in [2, 3]:
            raise Exception('State matrix must be a 2D or 3D array.')
        if B.ndim not in [2, 3]:
            raise Exception('Control-input matrix must be a 2D or 3D array.')

        if A.ndim == 2:
            if A.shape[0] != A.shape[1]:
                raise Exception(
                    'State matrix must be square along lag dimension.')
            A = np.expand_dims(A, axis=0)

        if B.ndim == 2:
            B = np.expand_dims(B, axis=0)

        if A.shape[1] != B.shape[1]:
            raise Exception('State and Control-input matrices must have ' +
                            'equal state rank')

        self.A = A.copy()
        self.rank_state = A.shape[2]
        self.lag_state = A.shape[0]

        self.B = B.copy()
        self.rank_exog = B.shape[2]
        self.lag_exog = B.shape[0]

        self.form = 'ord_p'

    def as_ord_1(self):
        """Convert from ord_p to ord_1."""

        if self.form == 'ord_1':
            return None

        l_S = self.lag_state
        l_E = self.lag_exog

        k_S = self.rank_state
        k_E = self.rank_exog

        AA = np.eye(k_S * l_S, k=-k_S)
        AA[:k_S, :] = self.A.transpose((1, 0, 2)).reshape(k_S, -1)

        BB = np.zeros((k_S * l_S, k_E * l_E))
        BB[:k_S, :] = self.B.transpose((1, 0, 2)).reshape(k_S, -1)

        self.A = AA
        self.B = BB
        self.form = 'ord_1'

    def as_ord_p(self):
        """Convert from ord_1 to ord_p."""

        if self.form == 'ord_p':
            return None

        l_S = self.lag_state
        l_E = self.lag_exog

        k_S = self.rank_state
        k_E = self.rank_exog

        assert self.A.shape == (k_S * l_S, k_S * l_S)
        k_S_hat = self.A.shape[0] / l_S
        assert k_S_hat == np.round(k_S_hat)
        k_S_hat = int(np.round(k_S_hat))

        AA = self.A[:k_S_hat, :].reshape(k_S_hat, -1, k_S_hat).transpose(
            1, 0, 2)

        assert self.B.shape == (k_S * l_S, k_E * l_E)
        k_S_hat = self.B.shape[0] / l_S
        assert k_S_hat == np.round(k_S_hat)
        k_S_hat = int(np.round(k_S_hat))

        k_E_hat = self.B.shape[1] / l_E
        assert k_E_hat == np.round(k_E_hat)
        k_E_hat = int(np.round(k_E_hat))

        BB = self.B[:k_S_hat, :].reshape(k_S_hat, -1, k_E_hat).transpose(
            1, 0, 2)

        self.A = AA
        self.B = BB
        self.form = 'ord_p'

    def get_eigs(self):
        """Return the eigenvalues of the system."""

        form = self.form
        self.as_ord_1()
        eigvals = np.linalg.eig(self.A)[0]

        if form != self.form:
            self.as_ord_p()

        return eigvals

    def is_stable(self):
        """Check stability of the state transition matrix."""

        eigvals = self.get_eigs()

        eig_max = np.abs(eigvals).max()
        if eig_max < 1:
            return True
        else:
            return False

    def schur_stabilize(self):
        """Force schur stability of the system."""

        form = self.form
        self.as_ord_1()
        self.A /= (np.abs(np.linalg.svd(self.A)[1]).max() + 1)

        if form != self.form:
            self.as_ord_p()

    def conv_state_to_lagged(self, X):
        """Make an auxilliary state matrix lagged based on LDS parameters."""

        # X has shape: [rank, N]
        if X.ndim != 2:
            raise Exception('X must be a 2-D array.')

        K, N = X.shape
        if K != self.rank_state:
            raise Exception(
                'Axis 0 of X must be of length equal to ' +
                'state-transition rank ({}).'.format(self.rank_state))

        if N < self.lag_state:
            raise Exception(
                'Axis 1 of X must have greater samples than ' +
                'lag-order of state-transition ({})'.format(self.lag_state))

        # Shift X to shape (L, K, N-L+1)
        # Return an unfolded array (K*L, N-L+1)
        X = np.array([
            X[:, l:(N - (self.lag_state - l) + 1)]
            for l in range(self.lag_state - 1, -1, -1)
        ])

        return X.reshape(-1, N - self.lag_state + 1)

    def conv_state_to_unlagged(self, X):
        """Make an auxilliary state matrix unlagged based on LDS parameters."""

        # X has shape: [rank*lags, N-rank]
        if X.ndim != 2:
            raise Exception('X must be a 2-D array.')

        KL, N = X.shape
        K = KL / self.lag_state

        if (KL != self.rank_state * self.lag_state) or (K != np.round(K)):
            raise Exception(
                'Axis 0 of X must be of length equal to ' +
                'state-transition rank ({}) '.format(self.rank_state) +
                'times state-transition lag ({}).'.format(self.lag_state))
        K = int(np.round(K))

        if self.lag_state == 1:
            return X
        else:
            X = X.reshape(self.lag_state, K, N)

            # Reconstruct the shortened signal
            # New X has shape: [K, N+lags-1]
            X = np.concatenate(
                (X[-1, :, :], X[0, :, 1 - self.lag_state:]), axis=1)

            return X
