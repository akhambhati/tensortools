import matplotlib.pyplot as plt
import numpy as np
import tensortools as tt

# Make synthetic dataset.
W, H, K = 4, 20, 4
WW = 1e-6*np.eye(K)

# Generate a state-transition matrix
dA = tt.dynamics.LDS(np.random.rand(K, K))
dA.schur_stabilize()
dA.as_ord_1()

# Propagate dynamical model
X = []
HH = [np.random.rand(K, 1)]
for i in range(H):
    X.append(WW.dot(HH[-1]))

    HH.append(dA.A.dot(HH[-1]))
X = np.array(X)[:, :, 0]
HH = np.array(HH)[:-1, :, 0]

# Normalize
X /= np.linalg.norm(X)

# Fit CP tensor decomposition (two times).
#model = tt.ncp_nnlds.init_model(X, rank=K, REG_dict=None, LDS_dict=None)
model = tt.ncp_nnlds.init_model(
    X,
    rank=K,
    REG_dict= None, #{'axis': 1,
              #'l1_ratio': 0.1,
              #'alpha': 1e-8},
    LDS_dict={
        'axis': 1,
        'beta': 2,
        'lags': 1,
        'init': 'rand'
    })

model = tt.ncp_nnlds.model_update(
    X, model, fit_dict={
        'min_iter': 10000,
        'max_iter': 10000,
        'verbose': True
    })
