import matplotlib.pyplot as plt
import numpy as np
import tensortools as tt
from scipy.stats import pearsonr

# Make synthetic dataset.
T = 10000
K = 3

# Generate an observation/mixing matrix
#WW = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0],
#               [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
WW = np.random.rand(5, 3)
print(WW.shape)

# Generate a state-transition matrix
dA = tt.dynamics.LDS(
    np.array([[0.99, 0.0, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 0.99]]))
#dA = tt.dynamics.LDS(np.random.rand(3, 3))
dA.schur_stabilize()
dA.as_ord_1()

# Propagate dynamical model
X = []
HH = [5 * np.abs(np.random.random(size=(K, 1)))]
for i in range(T):
    n1 = 0 * np.abs(np.random.rand(WW.shape[0], 1))
    n2 = 1e-4 * np.abs(np.random.rand(K, 1))
    X.append(WW.dot(HH[-1]) + n1)
    HH.append(dA.A.dot(HH[-1]) + n2)
X = np.array(X)[:, :, 0]
HH = np.array(HH)[:-1, :, 0]

# Cut-off the transient simulation
train_ix = slice(1000, 8000)
test_ix = slice(8000, T)

# Fit CP tensor decomposition (two times).
# Initialize model
model = tt.ncp_nnlds.init_model(
    X[train_ix, :] / np.linalg.norm(X[train_ix, :]),
    rank=3,
    NTF_dict={'beta': 2,
              'init': 'rand'},
    REG_dict={'axis': 0,
              'l1_ratio': 0.0,
              'alpha': 1e0},
    LDS_dict={'axis': 0,
              'beta': 2,
              'lags': 1,
              'init': 'rand'},
    random_state=None)

model = tt.ncp_nnlds.model_update(
    X[train_ix, :] / np.linalg.norm(X[train_ix, :]),
    model,
    fit_dict={
        'min_iter': 1,
        'max_iter': np.inf,
        'tol': 1e-6,
        'verbose': True
    })

# Map factor to
map_table = np.corrcoef(HH[train_ix, :].T,
                        model.model_param['NTF']['W'].factors[0].T)[K:, :K]
map_fac = np.argmax(map_table, axis=1)

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(K):
    ax = plt.subplot(3, 1, i + 1)
    ax.plot(HH[train_ix, i] / HH[train_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        model.model_param['NTF']['W'].factors[0][:, map_fac[i]] /
        model.model_param['NTF']['W'].factors[0][:, map_fac[i]].max(),
        color='r',
        linewidth=0.25)
    ax.legend(['True', 'Fitted'])
    ax.set_title(
        pearsonr(model.model_param['NTF']['W'].factors[0][:, map_fac[i]],
                 HH[train_ix, i]))
plt.show()

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(5):
    ax = plt.subplot(5, 1, i + 1)
    ax.plot(X[train_ix, i] / X[train_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        model.model_param['NTF']['W'].full()[:, i] /
        model.model_param['NTF']['W'].full()[:, i].max(),
        color='r',
        linewidth=0.25)
    ax.legend(['True', 'Fitted'])
    ax.set_title(
        pearsonr(model.model_param['NTF']['W'].full()[:, i], X[train_ix, i]))
plt.show()

# Forecast
XP = tt.ncp_nnlds.model_forecast(
    X[test_ix, :] / np.linalg.norm(X[train_ix, :]),
    model,
    forecast_steps=1,
    fit_dict={
        'min_iter': 1,
        'max_iter': np.inf,
        'tol': 1e-6,
        'verbose': False
    })

# Map factor to
map_table = np.corrcoef(HH[test_ix, :].T, XP[0].factors[0].T)[K:, :K]
map_fac = np.argmax(map_table, axis=1)

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(K):
    ax = plt.subplot(3, 1, i + 1)
    ax.plot(HH[test_ix, i] / HH[test_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        XP[0].factors[0][:, map_fac[i]] /
        XP[0].factors[0][:, map_fac[i]].max(),
        color='r',
        linewidth=0.25)
    ax.legend(['True', 'Fitted'])
    ax.set_title(pearsonr(XP[0].factors[0][:, map_fac[i]], HH[test_ix, i]))
plt.show()

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(5):
    ax = plt.subplot(5, 1, i + 1)
    ax.plot(X[test_ix, i] / X[test_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        XP[0].full()[:, i] / XP[0].full()[:, i].max(),
        color='r',
        linewidth=0.25)
    ax.legend(['True', 'Fitted'])
    ax.set_title(pearsonr(XP[0].full()[:, i], X[test_ix, i]))
plt.show()
