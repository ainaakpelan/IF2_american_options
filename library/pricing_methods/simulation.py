import numpy as np
import math

def simulate_gbm_paths(
    S0: np.ndarray,
    r: float,
    sigma: np.ndarray,
    q: np.ndarray,
    T: float,
    M: int,
    N: int,
    corr: np.ndarray = None,
    seed: int = None,
    antithetic: bool = False
) -> np.ndarray:
    """
    Simulate N Monte Carlo GBM paths with continuous dividend yields, 
    optionally correlated across d assets.

    Parameters
    ----------
    S0 : array-like, shape (d,)
        Initial asset prices.
    r : float
        Risk-free rate.
    sigma : array-like, shape (d,)
        Volatilities.
    q : array-like, shape (d,)
        Continuous dividend yields.
    T : float
        Time to maturity.
    M : int
        Number of time steps.
    N : int
        Number of Monte Carlo paths (if antithetic=True, must be even).
    corr : array-like, shape (d, d), optional
        Correlation matrix. If None, assets are simulated independently.
    seed : int or None, optional
        Random seed.
    antithetic : bool, default False
        Whether to use antithetic variates.

    Returns
    -------
    paths : ndarray, shape (N, M+1, d)
        Simulated asset-price paths.
    """
    S0    = np.asarray(S0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q     = np.asarray(q, dtype=float)
    d     = S0.shape[0]

    if sigma.shape[0] != d or q.shape[0] != d:
        raise ValueError("sigma and q must have same length as S0")

    if corr is None:
        L = np.eye(d)
    else:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (d, d):
            raise ValueError("corr must be a (d, d) matrix")
        L = np.linalg.cholesky(corr)

    if seed is not None:
        np.random.seed(seed)

    dt        = T / M
    drift_dt  = (r - q - 0.5 * sigma**2) * dt
    vol_sqrt  = sigma * math.sqrt(dt)

    paths = np.empty((N, M+1, d), dtype=float)
    paths[:, 0, :] = S0

    if antithetic:
        if N % 2 != 0:
            raise ValueError("N must be even when antithetic=True")
        half = N // 2

    for t in range(1, M+1):
        # generate standard normals
        if antithetic:
            Zh = np.random.normal(size=(half, d))
            Zh = Zh @ L.T
            Z  = np.vstack([Zh, -Zh])
        else:
            Z = np.random.normal(size=(N, d)) @ L.T

        paths[:, t, :] = paths[:, t-1, :] * np.exp(drift_dt + vol_sqrt * Z)

    return paths
