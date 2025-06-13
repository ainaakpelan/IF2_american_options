from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np

from .simulation import simulate_gbm_paths
from .data_structures import ModelParams

class AmericanOptionPricer(ABC):
    """
    Abstract base class for American option pricers.
    Supports multi-dimensional underlyings via Monte Carlo path simulation.
    Subclasses must implement the `price` method, and fill in
    `_cashflows`, `_execution_values` and `_decision_matrix`.
    """

    def __init__(
        self,
        model_params: ModelParams,
        payoff: Callable[[np.ndarray], np.ndarray],
        exercise_grid: Optional[np.ndarray] = None
    ):
        """
        Initialize the pricer.

        Parameters
        ----------
        model_params : ModelParams
            Market and simulation parameters, including:
            - spot          : np.ndarray of shape (d,), initial asset prices
            - rf_rate       : float, risk-free interest rate
            - sigma         : np.ndarray of shape (d,), volatilities
            - corr          : np.ndarray of shape (d,d), correlation matrix
            - dividend_yield: np.ndarray of shape (d,), continuous dividend yields
            - maturity      : float, time to expiry
            - n_steps       : int, number of time steps
            - n_paths       : int, number of Monte Carlo paths
        payoff : Callable
            A function that takes simulated paths of shape
            (n_paths, n_steps+1, d) and returns immediate payoffs
            as a (n_paths, n_steps+1) array.
        exercise_grid : Optional[np.ndarray]
            Times at which early exercise is allowed. If None,
            a uniform grid over [0, maturity] is used.
        """
        self.model_params = model_params
        self.payoff = payoff

        # unpack model parameters
        self.S0 = np.asarray(model_params.spot)
        self.r = model_params.rf_rate
        self.sigma = np.asarray(model_params.sigma)
        self.corr = np.asarray(model_params.corr)
        self.dividend_yield = np.asarray(model_params.dividend_yield)
        self.T = model_params.maturity
        self.n_steps = model_params.n_steps
        self.n_paths = model_params.n_paths

        # time grid
        self.time_grid = np.linspace(0.0, self.T, self.n_steps + 1)
        self.dt = self.T / self.n_steps

        # exercise times
        self.exercise_grid = (np.asarray(exercise_grid)
                              if exercise_grid is not None
                              else self.time_grid)

        # placeholders for results
        self._paths: Optional[np.ndarray] = None
        self._cashflows: Optional[np.ndarray] = None
        self._execution_values: Optional[np.ndarray] = None
        self._decision_matrix: Optional[np.ndarray] = None

    def _simulate_paths(self) -> np.ndarray:
        """
        Simulate geometric Brownian motion paths with correlation
        and continuous dividend yields.

        Returns
        -------
        paths : np.ndarray
            Simulated asset paths of shape (n_paths, n_steps+1, d).

        Notes
        -----
        - In the correlated version (`simulate_paths_gbm`), we now
          pass `dividend_yield` so that drift = r - q is applied.
        - In the fallback (`simulate_gbm_paths`), we also include
          dividends via drift = r - q.
        """
        try:
            paths = simulate_paths_gbm(
                S0=self.S0,
                r=self.r,
                sigma=self.sigma,
                corr=self.corr,
                q=self.dividend_yield,
                T=self.T,
                M=self.n_steps,
                N=self.n_paths
            )
        except ImportError:
            # fallback to independent GBM with dividend yields included
            cloud = simulate_gbm_paths(
                spot_prices=self.S0,
                drift=(self.r - self.dividend_yield),
                vol=self.sigma,
                n_time_pts=self.n_steps,
                n_paths=self.n_paths,
                max_time=self.T
            )
            # cloud: (d, n_paths, n_steps+1) → reshape to (n_paths, n_steps+1, d)
            paths = np.transpose(cloud, (1, 2, 0))

        self._paths = paths
        return paths

    @abstractmethod
    def price(self) -> float:
        """
        Calculate the American option price.

        Subclasses must:
        1. call self._simulate_paths()
        2. compute self._execution_values (intrinsic values)
        3. decide early exercise → self._decision_matrix
        4. compute discounted cashflows → self._cashflows
        5. return the Monte Carlo estimate of the option price
        """
        ...

    def get_cashflows(self) -> np.ndarray:
        """
        Return simulated cashflows for each path and time.

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps+1)
        """
        return self._cashflows

    def get_execution_values(self) -> np.ndarray:
        """
        Return intrinsic (execution) values per path and time.

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps+1)
        """
        return self._execution_values

    def get_decision_matrix(self) -> np.ndarray:
        """
        Return the exercise decision matrix (0 or 1) per path and time.

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps+1)
        """
        return self._decision_matrix
