import numpy as np
import pandas as pd
import os
from loguru import logger as log

ADF_CRIT_VALUES = pd.read_csv(os.path.join(".", "algo_model", "tables", "ADF_CRIT_VALUES.csv"), index_col="var")

def adf_crit_value(p: float, N: int, model: str):
    """
    M0: No drift and no trend
    M1: Drift and no trend
    M2: Drift and trend
    """
    allowed_models = ["M0", "M1", "M2"]
    assert model in allowed_models
    allowed_p_vals = [0.01, 0.025, 0.05, 0.1]
    assert p in allowed_p_vals
    col_name = f"{model}: {p}"
    [t, u, v, w] = ADF_CRIT_VALUES[col_name]
    crit = t + u/N + v/(N**2) + w/(N**3)
    return crit

def dickey_fuller(ts: np.ndarray):
    # Assumes timeseries is AR1, of the form y_t = β_1 + Φ_1 * y_{t-1} + ε_t
    # Cases H_0: Φ_1 = 1 (unit root -> not stationary)
    #       H_1: Φ_1 < 1  (no unit root -> stationary)
    # Process: Subtract previous timestep
    #       y_t - y_{t-1} = β_1 + (Φ_1 - 1) * y_{t-1} + ε_t
    #       Δy_t = β_1 + δ * y_{t-1} + ε_t
    if len(ts.shape) == 1:
        ts = ts[:, np.newaxis]
    assert len(ts.shape) == 2 and ts.shape[1] == 1
    y_prev = ts[0:-1]
    y_curr = ts[1:]
    delta_y = y_curr - y_prev
    N = delta_y.shape[0]
    assert N > 1
    # Note that the null & alternative hypotheses can be rewritten as
    #       H_0: δ = 0
    #       H_1: δ < 0
    # Where, when H_0 is assumed the equation simplifies to Δy_t = β_1 + ε_t
    # Now evaluate the t-statistic of δ by OLS
    #       OLS of the form y = mx + b where Δy -> y, y_{t-1} -> x
    #       Keep it in matrix form: β = (X'X)^-1 X'y, where β = [[b], [m]]
    #       Note 'hat' means the variable is an estimator
    X = np.array([np.ones(N), y_prev[:,0]], dtype=np.float64).T
    X2_inv = np.linalg.inv(X.T @ X)
    beta_hat = X2_inv @ X.T @ delta_y
    y_hat = X @ beta_hat
    sample_var = np.sum(np.square(y_hat - delta_y)) / (N-2)
    var_beta_hat = np.diag(sample_var * X2_inv)[:, np.newaxis]
    std_err_beta_hat = np.sqrt(var_beta_hat)

    t_statistic = beta_hat[1,0]/std_err_beta_hat[1,0]
    
    log.info(f"Calculated t-statistic of {t_statistic}")
    return t_statistic
