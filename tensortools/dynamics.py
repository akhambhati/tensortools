"""
Class to instantiate a Linear Dynamical System model

Author: Ankit N. Khambhati
Last Updated: 2018/11/14
"""
import numpy as np


class LDS(object):
    """Linear Dynamical System object.

    Attributes
    ----------
    A : np.ndarray; shape: [lag x rank x rank]
        State matrix containing transition coefficients.
    lag : int
        Number of time-lags represented by the dynamical model.
    rank : int
        Rank of the system (equivalently the number of states).
    form : ['ord_1', 'ord_p']
        Specifies the current formulation of the state matrix.
        A p-th order model can always be represented as a 1-st order model.
    """

    def __init__(self, A):
        """Initializes LDS.

        Parameters
        ----------
        A : np.ndarray; shape: [n_lag x n_rank x n_rank]
            State matrix (in ord_p form) in containing transition coefficients.
        """

        if A.ndim not in [2, 3]:
            raise Exception('State matrix must be a 2D or 3D array.')

        if A.ndim == 2:
            if A.shape[0] != A.shape[1]:
                raise Exception(
                    'State matrix must be square along lag dimension.')
            A = np.expand_dims(A, axis=0)

        self.A = A.copy()
        self.shape = self.A.shape
        self.lag = self.shape[0]
        self.rank = self.shape[1]
        self.form = 'ord_p'

    def as_ord_1(self):
        """Convert from ord_p to ord_1."""

        if self.form == 'ord_1':
            return None

        l = self.lag
        k = self.rank

        AA = np.eye(k * l, k=-k)
        AA[:k, :] = self.A.transpose((1, 0, 2)).reshape(k, -1)

        self.A = AA
        self.shape = self.A.shape
        self.form = 'ord_1'

    def as_ord_p(self):
        """Convert from ord_1 to ord_p."""

        if self.form == 'ord_p':
            return None

        kl, _ = self.A.shape
        k = kl / self.lag
        assert k == np.round(k)
        k = int(np.round(k))

        AA = self.A[:k, :].reshape(k, -1, k).transpose(1, 0, 2)

        self.A = AA
        self.shape = self.A.shape
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

    def conv_X_to_lagged(self, X):
        """Make an auxilliary state matrix lagged based on LDS parameters."""

        # X has shape: [rank, N]
        if X.ndim != 2:
            raise Exception('X must be a 2-D array.')

        K, N = X.shape
        if K != self.rank:
            raise Exception('Axis 0 of X must be of length equal to ' +
                            'LDS rank ({}).'.format(self.rank))

        if N < self.lag:
            raise Exception('Axis 1 of X must have greater samples than ' +
                            'lag-order of LDS ({})'.format(self.lag))

        # Shift X to shape (L, K, N-L+1)
        # Return an unfolded array (K*L, N-L+1)
        X = np.array([
            X[:, l:(N - (self.lag - l) + 1)]
            for l in range(self.lag - 1, -1, -1)
        ])

        return X.reshape(-1, N - self.lag + 1)

    def conv_X_to_unlagged(self, X):
        """Make an auxilliary state matrix unlagged based on LDS parameters."""

        # X has shape: [rank*lags, N-rank]
        if X.ndim != 2:
            raise Exception('X must be a 2-D array.')

        KL, N = X.shape
        K = KL / self.lag

        if (KL != self.rank * self.lag) or (K != np.round(K)):
            raise Exception('Axis 0 of X must be of length equal to ' +
                            'LDS rank ({}) times LDS lag ({}).'.format(
                                self.rank, self.lag))
        K = int(np.round(K))

        if self.lag == 1:
            return X
        else:
            X = X.reshape(self.lag, K, N)

            # Reconstruct the shortened signal
            # New X has shape: [K, N+lags-1]
            X = np.concatenate((X[-1, :, :], X[0, :, 1 - self.lag:]), axis=1)

            return X

    def __getitem__(self, p):
        """Gets state matrix for (p-1)-th lag."""
        return self.A[p]
