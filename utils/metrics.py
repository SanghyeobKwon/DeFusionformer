import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def nRMSE(pred, true):
    return np.mean(np.abs((pred - true) / (np.max(true)-np.min(true))))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    nrmse = nRMSE(pred, true)

    return mae, mse, rmse, nrmse
