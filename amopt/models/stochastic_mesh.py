import numpy as np
from scipy.stats import norm


def simulate_gbm_paths(spot_prices, drift, vol, n_time_pts, n_paths, max_time):
    """
    Simulates multiple independent GBM paths.

    Parameters:
        spot_prices (array-like): Initial values for each GBM (length d)
        mu (array-like): Drifts for each GBM (length d)
        sigma (array-like): Volatilities for each GBM (length d)
        n_paths (int): Number of paths per GBM
        time_grid (array-like): Time steps, shape (m,)

    Returns:
        gbms (ndarray): Array of shape (K, N, m)
    """
    dimension = len(spot_prices)
    if len(drift) != dimension or len(vol) != dimension:
        raise ValueError(
            "mu and sigma must have the same length as spot_prices (d-dimensional GBM)"
        )
    time_grid = np.linspace(0, max_time, n_time_pts + 1)
    dt = max_time / n_time_pts

    # Output array: (d, b, m) == (dimension, n_paths, n_time_pts + 1)
    gbms = np.zeros((dimension, n_paths, n_time_pts + 1))

    for k in range(dimension):
        Z = np.random.normal(size=(n_paths, n_time_pts + 1))
        dW = np.sqrt(dt) * Z
        dW[:, 0] = 0
        W = np.cumsum(dW, axis=1)
        exponent = (drift[k] - 0.5 * vol[k] ** 2) * time_grid + vol[k] * W
        gbms[k] = spot_prices[k] * np.exp(exponent)
    # TODO add antithetic variates
    return gbms


def likelihood_weights(x, y, vol, rf_rate, dividend_yield, dt):
    """
    Computes the function f(x, y) as defined in the GBM transition density.

    Parameters:
        x (ndarray): d-dimensional array (shape: (d,))
        y (ndarray): d-dimensional array (shape: (d,))
        vol (float): d-dimensional array of volatilities of single assets (shape: (d,))
        rf_rate (float): risk-free rate
        delta (ndarray): d-dimensional array of dividend yields of single assets (shape: (d,))
        dt (float): time step

    Returns:
        float: value of f(x, y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    vol = np.asarray(vol)
    dividend_yield = np.asarray(dividend_yield)

    if x.shape != y.shape or vol.shape != x.shape or dividend_yield.shape != x.shape:
        raise ValueError(
            "x, y, vol, and dividend_yield arrays must have the same shape"
        )

    # Compute the argument of the normal PDF
    numerator = np.log(y / x) - (rf_rate - dividend_yield - 0.5 * vol**2) * dt
    denominator = vol * np.sqrt(dt)
    z = numerator / denominator

    # Compute the scaled normal PDF values
    normal_pdf_vals = norm.pdf(z)
    density_terms = normal_pdf_vals / (vol * y * np.sqrt(dt))

    return np.prod(density_terms)


def mesh_estimator(asset_price_mesh, payoff_func, vol, rf_rate, dividend_yield, dt):
    """
    Performs the stochastic mesh backward induction.

    Parameters:
        X: np.ndarray of shape (d, b, m) — simulated asset paths
        payoff_fn: callable — h(x): payoff function from R^d to R
        f_xy: callable — transition density f(x, y)
        sigma, r, delta, dt: parameters for f_xy

    Returns:
        V: np.ndarray of shape (b, m) — estimated value function
    """
    d, b, m = asset_price_mesh.shape
    option_prices = np.zeros((b, m))  # m is the number of time steps including zero

    disc_fact = np.exp(-rf_rate * dt)

    # Instant payoff for every point
    instant_payoff = payoff_func(asset_price_mesh)
    instant_payoff = np.reshape(instant_payoff, (b, m))
    # Initialize last column with payoffs
    option_prices[:, -1] = instant_payoff[:, -1]

    # Backward induction
    for i in reversed(range(1, m - 1)):
        option_prices_next = option_prices[:, i + 1]
        for j in range(b):
            weights = np.array(
                [
                    likelihood_weights(
                        asset_price_mesh[:, j, i],  # X_{ij}
                        asset_price_mesh[:, k, i + 1],  # X_{i+1, k} for all k's
                        vol,
                        rf_rate,
                        dividend_yield,
                        dt,
                    )
                    for k in range(b)
                ]
            )
            option_prices[j, i] = np.maximum(
                disc_fact * np.mean(weights * option_prices_next),
                instant_payoff[j, i],
            )
    option_prices[:, 0] = disc_fact * np.mean(option_prices[:, 1])

    return option_prices


def path_estimator(
    asset_price_mesh,
    payoff_func,
    spot_prices,
    drift,
    vol,
    rf_rate,
    dividend_yield,
    max_time,
    dt,
    n_paths=500,
):
    disc_fact = np.exp(-rf_rate * dt)
    d, b, m = asset_price_mesh.shape
    new_asset_cloud = simulate_gbm_paths(
        spot_prices, drift, vol, m - 1, n_paths, max_time
    )
    instant_payoff = payoff_func(new_asset_cloud)
    instant_payoff = np.reshape(instant_payoff, (n_paths, m))
    option_prices = mesh_estimator(
        asset_price_mesh, payoff_func, vol, rf_rate, dividend_yield, dt
    )
    # Continuation values based on the mesh
    continuation_values = np.zeros(new_asset_cloud.shape[1:])
    # Backward induction - continuation values only
    for i in reversed(range(1, m - 1)):  # as many time steps as in the mesh
        option_prices_next = option_prices[:, i + 1]
        for j in range(n_paths):
            weights = np.array(
                [
                    likelihood_weights(
                        new_asset_cloud[:, j, i],  # x
                        asset_price_mesh[:, k, i + 1],  # X_{i+1, k} for all k's
                        vol,
                        rf_rate,
                        dividend_yield,
                        dt,
                    )
                    for k in range(b)
                ]
            )
            continuation_values[j, i] = disc_fact * np.mean(
                weights * option_prices_next
            )
    excercise_moments = (instant_payoff >= continuation_values) | (
        np.abs(instant_payoff - continuation_values) < 1e-10
    )
    mask = excercise_moments[:, 1:]  # we do not let the option to be exercised at t=0
    stopping_times = np.argmax(mask, axis=1)
    # Reflect the fact that the stopping times were shifted by one
    stopping_times[mask.any(axis=1)] += 1
    # In case of no exercise, set the stopping time to maturity
    stopping_times[~mask.any(axis=1)] = m - 1

    return np.mean(
        np.exp(-rf_rate * dt * stopping_times)
        * instant_payoff[np.arange(n_paths), stopping_times]
    )


def price_option(
    asset_price_mesh,
    payoff_func,
    spot_prices,
    drift,
    vol,
    rf_rate,
    dividend_yield,
    max_time,
    dt,
    n_paths=500,
):
    """
    Wrapper of stochastic_mesh function to compute the option price at t=0 using stochastic mesh method.
    """
    option_prices = mesh_estimator(
        asset_price_mesh, payoff_func, vol, rf_rate, dividend_yield, dt
    )
    high_price = option_prices[0, 0]

    low_price = path_estimator(
        asset_price_mesh,
        payoff_func,
        spot_prices,
        drift,
        vol,
        rf_rate,
        dividend_yield,
        max_time,
        dt,
        n_paths,
    )

    return low_price, high_price
