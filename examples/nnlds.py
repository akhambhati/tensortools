import matplotlib.pyplot as plt
import numpy as np
import tensortools as tt
from scipy.stats import pearsonr

# Make synthetic dataset.
T = 10000
K_s = 2
K_e = 1
lag_state = 4
lag_exog = 1

# Generate an observation/mixing matrix
WW = 0.5 * np.eye(K_s)

# Generate a state-transition matrix
A = np.random.rand(lag_state, K_s, K_s)
A = np.array([a**(i + 1) for i, a in enumerate(A)])
B = np.random.rand(lag_exog, K_s, K_e)
dAB = tt.LDS(A, B)
dAB.as_ord_1()
dAB.A /= np.linalg.norm(dAB.A)
dAB.as_ord_p()

# Generate dynamics
HH = [5 * np.abs(np.random.random((K_s, 1))) for ll in range(lag_state)]
UU = np.random.binomial(1, p=0.25, size=(K_e, T))
for i in range(T - len(HH)):
    H_ix = range(len(HH) - 1, len(HH) - 1 - lag_state, -1)
    AX = np.array([dAB.A[ii, :, :].dot(HH[ij])
                   for ii, ij in enumerate(H_ix)]).sum(axis=0)

    U_ix = range(len(HH) - 1, len(HH) - 1 - lag_exog, -1)
    BU = np.array([
        dAB.B[ii, :, :].dot(UU[:, [ij]]) for ii, ij in enumerate(U_ix)
    ]).sum(axis=0)

    HH.append(AX + BU)
HH = np.array(HH)[:, :, 0]
XX = WW.dot(HH.T)

# Train Model
train_ix = slice(0, int(2 / 3.0 * T))
test_ix = slice(int(2 / 3.0 * T), T)

# Fit CP tensor decomposition (two times).
# Initialize model
model = tt.ncp_nnlds.init_model(
    XX.T[train_ix, :],
    rank=K_s,
    NTF_dict={'beta': 2,
              'init': 'rand'},
    LDS_dict={
        'axis': 0,
        'beta': 2,
        'lag_state': lag_state,
        'lag_exog': lag_exog,
        'init': 'rand'
    },
    exog_input=UU[:, train_ix].T,
    random_state=None)

# Fix W
model.model_param['NTF']['W'][1] = WW.copy()

model = tt.ncp_nnlds.model_update(
    XX.T[train_ix, :],
    model,
    exog_input=UU[:, train_ix].T,
    fixed_axes=[1],
    fit_dict={
        'min_iter': 1,
        'max_iter': 1000,
        'tol': 1e-6,
        'verbose': True
    })

# Map factor to
map_table = np.corrcoef(HH[train_ix, :].T,
                        model.model_param['NTF']['W'].factors[0].T)[K_s:, :K_s]
map_fac = np.argmax(map_table, axis=1)

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(K_s):
    ax = plt.subplot(K_s, 1, i + 1)
    ax.plot(HH[train_ix, i] / HH[train_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        model.model_param['NTF']['W'].factors[0][:, map_fac[i]] /
        model.model_param['NTF']['W'].factors[0][:, map_fac[i]].max(),
        color='b',
        linewidth=0.25)
    ax.plot(
        0.5 * UU[0, train_ix] / UU[0, train_ix].max(),
        color='r',
        linewidth=0.1)
    ax.legend(['True', 'Fitted', 'Stim'])
    ax.set_title(
        pearsonr(model.model_param['NTF']['W'].factors[0][:, map_fac[i]],
                 HH[train_ix, i]))
plt.show()

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(K_s):
    ax = plt.subplot(K_s, 1, i + 1)
    ax.plot(
        XX.T[train_ix, i] / XX.T[train_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(
        model.model_param['NTF']['W'].full()[:, i] /
        model.model_param['NTF']['W'].full()[:, i].max(),
        color='B',
        linewidth=0.25)
    ax.plot(
        0.5 * UU[0, train_ix] / UU[0, train_ix].max(),
        color='r',
        linewidth=0.1)
    ax.legend(['True', 'Fitted', 'Stim'])
    ax.set_title(
        pearsonr(model.model_param['NTF']['W'].full()[:, i],
                 XX.T[train_ix, i]))
plt.show()

# Forecast
XP = []
for ii in range(test_ix.start, test_ix.stop):
    XP.append(
        tt.ncp_nnlds.model_forecast(
            XX.T[ii - 100:ii, :],
            UU.T[ii - 100:ii + 1, :],
            model,
            fit_dict={
                'min_iter': 1,
                'max_iter': 100,
                'tol': 1e-6,
                'verbose': False
            }).full()[0, :])
XP = np.array(XP)

# Generate Plots Evaluating Fit
plt.figure(figsize=(8, 3), dpi=100)
for i in range(K_s):
    ax = plt.subplot(K_s, 1, i + 1)
    ax.plot(
        XX.T[test_ix, i] / XX.T[test_ix, i].max(), color='k', linewidth=0.5)
    ax.plot(XP[:, i] / XP[:, i].max(), color='B', linewidth=0.25)
    ax.plot(
        0.5 * UU.T[test_ix, 0] / UU.T[test_ix, 0].max(),
        color='r',
        linewidth=0.1)
    ax.legend(['True', 'Fitted', 'Stim'])
    ax.set_title(pearsonr(XP[:, i], XX.T[test_ix, i]))
plt.show()
