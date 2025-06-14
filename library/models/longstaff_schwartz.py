import numpy as np
import math
from itertools import product
import numpy as np
import math


def simulate_paths_gbm(S0, r, sigma, corr, T, M, N, seed=None, antithetic=False):
    """
    Simulate N paths of a d-dimensional GBM with correlation. Optionally use
    antithetic variates: if antithetic=True, N must be even; we generate N/2
    independent draws and then include their negations.

    Parameters
    ----------
    S0 : array-like, shape (d,)
        Initial asset prices.
    r : float
        Risk-free rate (continuous).
    sigma : array-like, shape (d,)
        Volatility per asset.
    corr : array-like, shape (d, d)
        Correlation matrix.
    T : float
        Time to maturity (years).
    M : int
        Number of time steps.
    N : int
        Total number of paths (must be even if antithetic=True).
    seed : int or None
        Random seed for reproducibility.
    antithetic : bool, default False
        Whether to use antithetic variates. If True, N must be even.

    Returns
    -------
    paths : np.ndarray, shape (N, M+1, d)
        Simulated asset‐price paths, including t=0.
    """
    S0 = np.asarray(S0)
    sigma = np.asarray(sigma)
    corr = np.asarray(corr)
    d = len(S0)

    if seed is not None:
        np.random.seed(seed)

    dt = T / M
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    L = np.linalg.cholesky(corr)

    # If antithetic, ensure N is even and generate N/2 draws + negatives
    if antithetic:
        if N % 2 != 0:
            raise ValueError("N must be even when antithetic=True")
        half_N = N // 2
        paths = np.empty((N, M + 1, d))
        paths[:, 0, :] = S0

        for t in range(1, M + 1):
            Z_half = np.random.normal(size=(half_N, d)) @ L.T  # (half_N, d)
            Z = np.vstack([Z_half, -Z_half])  # (N, d)
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + vol * Z)
    else:
        paths = np.empty((N, M + 1, d))
        paths[:, 0, :] = S0
        for t in range(1, M + 1):
            Z = np.random.normal(size=(N, d)) @ L.T
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + vol * Z)

    return paths


def _build_basis(S: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Construct a polynomial basis of total degree ≤ `degree` for regression,
    without cross‐terms between different assets. That is, for each asset j
    include S[:, j]**k for k = 1..degree, plus a constant column 1.

    Parameters
    ----------
    S : np.ndarray, shape (n, d)
        Current asset prices for the in-the-money paths:
        n = number of in–the–money paths, d = number of underlying assets.
    degree : int, default 2
        Maximum polynomial degree to include (k = 0..degree).

    Returns
    -------
    X : np.ndarray, shape (n, n_cols)
        Regression matrix with columns:
        [1]                                  (constant term)
        [S[:, 0], S[:, 0]**2, ..., S[:, 0]**degree]  (powers of asset 0)
        [S[:, 1], S[:, 1]**2, ..., S[:, 1]**degree]  (powers of asset 1)
        ...
        [S[:, d-1], S[:, d-1]**2, ..., S[:, d-1]**degree]
        Total number of columns = 1 + d * degree.
    """
    n, d = S.shape
    cols = [np.ones(n)]  # constant column
    # For each asset j, include all powers j^k for k=1..degree
    for j in range(d):
        for k in range(1, degree + 1):
            cols.append(S[:, j] ** k)
    return np.column_stack(cols)


def _build_basis_with_cross(S: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Construct a polynomial basis of total degree ≤ `degree`, including cross‐terms
    between different assets. That is, include every monomial
       S[:, 0]**e0 * S[:, 1]**e1 * ... * S[:, d-1]**e{d-1}
    for nonnegative exponents e0, e1, …, e{d-1} whose sum is between 0 and degree.
    In particular:
    - The constant term (all exponents = 0).
    - All pure powers S[:, j]**k for k=1..degree.
      - All products S[:, i]**a * S[:, j]**b for a+b ≤ degree, etc.

    Parameters
    ----------
    S : np.ndarray, shape (n, d)
        Asset prices at time t for the in–the–money paths.
    degree : int, default 2
        Maximum total degree of monomials to include.

    Returns
    -------
    X : np.ndarray, shape (n, n_cols)
        Regression matrix where each column is one monomial whose multi‐index
        exponents (e0, e1, …, e{d-1}) satisfy sum(ei) ≤ degree. The first column
        is the constant term where all ei=0.
    """
    n, d = S.shape

    multi_indices = [
        idx for idx in product(range(degree + 1), repeat=d) if sum(idx) <= degree
    ]

    # Build each monomial column by taking product of S[:, j]**e[j]
    cols = []
    for exponents in multi_indices:
        # Compute monomial vector for this exponent tuple
        mon = np.ones(n)
        for j in range(d):
            e_j = exponents[j]
            if e_j != 0:
                mon = mon * (S[:, j] ** e_j)
        cols.append(mon)

    # Stack into an (n, num_basis) array
    return np.column_stack(cols)


def _build_basis_laguerre(S: np.ndarray, K: float, degree: int = 2) -> np.ndarray:
    """
    Construct a univariate Laguerre polynomial basis up to a given degree for one-dimensional input.

    This function remains unchanged: it handles only the case d=1 and returns
    columns [L0(x), L1(x), ..., L_degree(x)], where


    X : np.ndarray, shape (n, degree + 1)
        Regression matrix whose columns are the Laguerre polynomials:
        - L0(x) = 1
        - L1(x) = 1 − x           (if degree ≥ 1)
        - L2(x) = 1 − 2x + 0.5 x² (if degree ≥ 2)
        - L[:, k] = ((2k - 1 - x) * L[:, k-1] - (k - 1) * L[:, k-2]) / k, for k ≥ 2

    Parameters
    ----------
    S : np.ndarray, shape (n, 1)
        Array of asset prices for the in-the-money paths. Only one column is allowed (d = 1).
    K : float
        Strike (or scaling) used to normalize S; compute x = S / K.
    degree : int, default 2
        Maximum degree of Laguerre polynomials to include (0..3).

    Returns
    -------
    X : np.ndarray, shape (n, degree + 1)
        Regression matrix, where column k is L_k(x).

    Raises
    ------
    AssertionError
        If `S` does not have exactly one column (i.e. d != 1).
    """
    n, d = S.shape
    assert d == 1, "Laguerre basis requires S to have shape (n, 1)."

    # 1) Compute x = S / K
    x = S[:, 0] / K  # shape (n,)

    # 2) Allocate array to hold L_0 through L_degree
    #    L_mat[:, k] will be L_k(x).
    L_mat = np.zeros((n, degree + 1), dtype=float)

    # 3) Base cases:
    #    L_0(x) = 1
    L_mat[:, 0] = 1.0

    if degree >= 1:
        #    L_1(x) = 1 - x
        L_mat[:, 1] = 1.0 - x

    # 4) Three‐term recurrence for k = 2..degree:
    #    L_k(x) = ((2k - 1 - x) * L_{k-1}(x) - (k - 1) * L_{k-2}(x)) / k
    for k in range(2, degree + 1):
        L_prev = L_mat[:, k - 1]  # L_{k-1}(x)
        L_prev2 = L_mat[:, k - 2]  # L_{k-2}(x)
        # Compute L_k(x) for all entries:
        L_mat[:, k] = ((2 * k - 1 - x) * L_prev - (k - 1) * L_prev2) / k

    return L_mat


import numpy as np
from itertools import product


def generate_multi_indices(d: int, p_max: int):
    """
    Generate all nonnegative integer tuples (k1, k2, …, kd) such that
    sum(k_i for i in range(d)) ≤ p_max.
    """
    indices = []
    for ks in product(range(p_max + 1), repeat=d):
        if sum(ks) <= p_max:
            indices.append(ks)
    return indices


def _build_basis_laguerre_multid(
    S: np.ndarray, K_vec: list[float] | np.ndarray, p_max: int = 2
) -> np.ndarray:
    """
    Construct a multidimensional Laguerre polynomial basis up to total degree p_max,
    computing each univariate Laguerre polynomial up to p_max via recurrence.

    For each path i (i = 1..n) and each multi-index (k1, k2, …, kd) with sum ≤ p_max,
    compute the product:
      L_{k1}(S[i,0]/K_vec[0]) * L_{k2}(S[i,1]/K_vec[1]) * … * L_{kd}(S[i,d-1]/K_vec[d-1])

    The resulting regression matrix has one column per multi-index. This basis
    captures cross‐dependencies in a structured (orthogonal) way.

    Parameters
    ----------
    S : np.ndarray, shape (n, d)
        Matrix of in–the–money asset prices at time t:
        n = number of ITM paths, d = number of assets.
    K_vec : list or np.ndarray of length d
        Strike or scaling factor for each dimension: [K1, K2, …, Kd].
    p_max : int, default 2
        Maximum sum of Laguerre indices (total polynomial degree).

    Returns
    -------
    X : np.ndarray, shape (n, num_basis)
        Regression matrix where num_basis = number of multi-indices (k1,…,kd)
        satisfying k1 + … + kd ≤ p_max. Each column is the product of univariate
        Laguerre polynomials L_{k_j}(S[:, j]/K_vec[j]) across dimensions.
    """
    S = np.asarray(S)
    K_vec = np.asarray(K_vec)
    n, d = S.shape
    assert K_vec.shape[0] == d, "Length of K_vec must equal number of assets d"

    # 1) Precompute univariate Laguerre basis for each dimension j via direct recurrence:
    #    L_j_mat will have shape (n, p_max+1), where column k is L_k(x_j), x_j = S[:,j]/K_vec[j].
    L_mats = []
    for j in range(d):
        xj = S[:, j] / K_vec[j]  # shape (n,)
        # Allocate space for L_0 ... L_p_max
        L_j = np.zeros((n, p_max + 1), dtype=float)
        # Base case L_0(x) = 1
        L_j[:, 0] = 1.0
        if p_max >= 1:
            # Base case L_1(x) = 1 - x
            L_j[:, 1] = 1.0 - xj
        # Recurrence for k = 2..p_max:
        for k in range(2, p_max + 1):
            L_prev = L_j[:, k - 1]  # L_{k-1}(xj)
            L_prev2 = L_j[:, k - 2]  # L_{k-2}(xj)
            # L_k(x) = ((2k - 1 - x) * L_{k-1}(x) - (k - 1) * L_{k-2}(x)) / k
            L_j[:, k] = ((2 * k - 1 - xj) * L_prev - (k - 1) * L_prev2) / k
        L_mats.append(L_j)

    # 2) Generate all multi-indices (k1, …, kd) with sum ≤ p_max
    multi_idx = generate_multi_indices(d, p_max)

    # 3) For each multi-index, multiply the corresponding univariate columns
    cols = []
    for ks in multi_idx:
        col = np.ones(n, dtype=float)
        for j in range(d):
            kj = ks[j]
            col *= L_mats[j][:, kj]
        cols.append(col)

    # 4) Stack columns into final matrix (n, num_basis)
    return np.column_stack(cols)


def _weighted_laguerre_polynomials(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the first (n+1) weighted Laguerre polynomials for a 1D input array x.
    Weighted Laguerre L_n^w(x) = exp(-x/2) * L_n(x), where L_n(x) is the standard
    (physicists’) Laguerre polynomial of degree n. We use the recurrence:

        L_0(x) = 1
        L_1(x) = 1 - x
        For k >= 2:
            L_k(x) = ((2*k - 1 - x) * L_{k-1}(x) - (k - 1) * L_{k-2}(x)) / k

    Then multiply each L_k(x) by exp(-x/2).

    Parameters
    ----------
    x : np.ndarray, shape (m,)
        One-dimensional array of nonnegative values (e.g., x = S / K).
    n : int
        Maximum degree of the Laguerre polynomial to compute (so we return
        degrees 0 through n, inclusive).

    Returns
    -------
    mat : np.ndarray, shape (m, n+1)
        Matrix whose columns are [L_0^w(x), L_1^w(x), ..., L_n^w(x)], where
        L_k^w(x) = exp(-x/2) * L_k(x).
    """
    m = x.shape[0]
    # Prepare storage: each column j will be L_j(x) before weighting
    L = np.zeros((m, n + 1), dtype=float)
    # Unweighted Laguerre:
    L[:, 0] = 1.0
    if n >= 1:
        L[:, 1] = 1.0 - x
    for k in range(2, n + 1):
        # Recurrence: ( (2k - 1 - x) * L_{k-1} - (k - 1) * L_{k-2} ) / k
        L[:, k] = (((2 * k - 1 - x) * L[:, k - 1]) - ((k - 1) * L[:, k - 2])) / k

    # Now apply the weight exp(-x/2)
    w = np.exp(-0.5 * x).reshape(m, 1)  # shape (m,1)
    return L * w  # Broadcast so each column L[:, k] is multiplied by exp(-x/2)


def _build_basis_weighted_laguerre(
    S: np.ndarray, K: float, degree: int = 2
) -> np.ndarray:
    """
    Construct a weighted Laguerre basis of maximum degree `degree` for a one-dimensional state S.

    Each column is L_k^w(X) = exp(-X/2) * L_k(X), for k = 0..degree, where
    X = S[:, 0] / K and L_k(X) is the standard Laguerre polynomial of degree k.

    Parameters
    ----------
    S : np.ndarray, shape (n, 1)
        Array of asset prices (in-the-money paths). Must have exactly one column.
    K : float
        Strike or scaling factor used to compute X = S[:, 0] / K.
    degree : int, default 2
        Maximum polynomial degree. Will produce `degree+1` columns.

    Returns
    -------
    Xmat : np.ndarray, shape (n, degree+1)
        Regression matrix whose columns are [L_0^w(X), L_1^w(X), ..., L_degree^w(X)].
    """
    n, d = S.shape
    assert d == 1, "Weighted-Laguerre basis supports only d=1"
    # Scale prices
    X = S[:, 0] / K  # shape (n,)

    # Compute weighted Laguerre polynomials up to degree
    return _weighted_laguerre_polynomials(X, degree)


def lsm_price_multi(
    S0,
    r,
    sigma,
    corr,
    T,
    M,
    N,
    payoff_fn,
    basis_fn=None,  # default to monomial basis if None
    seed: int = 0,
    antithetic: bool = False,
):
    """
    American option pricing via Longstaff–Schwartz Monte Carlo, with optional
    antithetic variates and custom regression basis.

    Parameters
    ----------
    S0 : list[float]
        Initial prices (length d).
    r : float
        Risk-free rate (continuous).
    sigma : list[float]
        Volatility per asset (length d).
    corr : list[list[float]]
        Correlation matrix (d × d).
    T : float
        Time to maturity (years).
    M : int
        Number of exercise dates (time steps).
    N : int
        Number of Monte Carlo paths (must be even if antithetic=True).
    payoff_fn : callable
        Function S -> intrinsic payoffs, where S has shape (n, d).
    basis_fn : callable or None, default None
        Function S_itm -> regression matrix X, where S_itm has shape (n_itm, d)
        and X has shape (n_itm, n_basis). If None, uses monomial basis of degree=2.
    seed : int, default 0
        Random seed for reproducibility.
    antithetic : bool, default False
        Whether to use antithetic variates in path simulation.

    Returns
    -------
    float
        The estimated American-option price.
    """
    # Default regression basis = monomial of degree=2 if none provided
    if basis_fn is None:
        basis_fn = lambda S_itm: _build_basis(S_itm, degree=2)

    # 1) simulate paths, possibly with antithetic variates
    paths = simulate_paths_gbm(
        S0, r, sigma, corr, T, M, N, seed=seed, antithetic=antithetic
    )
    dt = T / M
    disc = math.exp(-r * dt)

    # 2) payoff at maturity
    cashflows = payoff_fn(paths[:, -1, :])

    # 3) backward induction
    for t in range(M - 1, 0, -1):
        S_t = paths[:, t, :]  # shape (N, d)
        intrinsic = payoff_fn(S_t)  # shape (N,)
        itm = intrinsic > 0  # boolean mask (N,)

        if np.any(itm):
            S_itm = S_t[itm]  # only in‐the‐money paths

            # build regression matrix via provided basis function
            X = basis_fn(S_itm)  # shape (n_itm, n_basis)

            Y = cashflows[itm] * disc  # discounted continuation
            coeff, *_ = np.linalg.lstsq(X, Y, rcond=None)
            continuation = X @ coeff

            exercise = intrinsic[itm] > continuation
            exercise_idx = np.where(itm)[0][exercise]
            cashflows[exercise_idx] = intrinsic[itm][exercise]

        cashflows *= disc

    return cashflows.mean() * disc


def payoff_put(K, idx=0):
    return lambda S: np.maximum(K - S[:, idx], 0.0)


def payoff_call(K, idx=0):
    return lambda S: np.maximum(S[:, idx] - K, 0.0)


def payoff_call_binary_conditional(K, idx_pay=0, idx_cond=1, H=1.0):
    return lambda S: np.maximum(S[:, idx_pay] - K, 0.0) * (S[:, idx_cond] > H)


def payoff_put_binary_conditional(K, idx_pay=0, idx_cond=1, H=1.0):
    return lambda S: np.maximum(K - S[:, idx_pay], 0.0) * (S[:, idx_cond] > H)
