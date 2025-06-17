import numpy as np
import time
from ..models.bjerksund_stensland import BjerksundStensland
from ..models.stochastic_mesh import price_option
from ..models.longstaff_schwartz import simulate_paths_gbm, lsm_price_multi

# from ..models.lsm import lsm_price_multi, simulate_paths_gbm


class AmericanOptionPricer:
    def __init__(
        self,
        spot_price,
        strike,
        risk_free_rate,
        volatility,
        time_to_maturity,
        execution_times,
        n_paths_mesh,
        corr=None,
        dividend_yield=0.0,
        seed=None,
        antithetic=True,
    ):
        self.spot_price = np.asarray(spot_price)
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.volatility = np.asarray(volatility)
        self.time_to_maturity = time_to_maturity
        self.dimension = len(spot_price) if isinstance(spot_price, (list, tuple)) else 1
        self.execution_times = np.asarray(execution_times)
        self.n_paths_mesh = n_paths_mesh
        self.corr = corr if corr is not None else np.identity(self.dimension)
        self.dividend_yield = np.asarray(dividend_yield)
        self.asset_paths = None
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.payoff_func = None
        self.antithetic = antithetic  # Default to True for antithetic variates
        # Prices placeholder
        self.lsm_price = None
        self.smm_mesh_price = None
        self.smm_path_price = None
        self.bjs_price = None

        # Validate parameters
        try:
            self._validate_parameters()
        except ValueError as e:
            raise ValueError(
                f"Invalid parameters for American option pricing: {e}"
            ) from e
        # Simulate asset paths
        self._simulate_asset_paths()

    def _simulate_asset_paths(self):
        # Simulate the asset paths for SMM & LSM
        self.asset_paths = simulate_paths_gbm(
            self.spot_price,
            self.risk_free_rate,
            self.volatility,
            self.corr,
            self.time_to_maturity,
            len(self.execution_times),
            self.n_paths_mesh,
            self.seed,
            self.antithetic,
        )

    def _validate_parameters(self):
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.dimension < 1:
            raise ValueError(
                "Spot price must be a positive number or a non-empty list/tuple"
            )
        if self.dividend_yield < 0:
            raise ValueError("Dividend yield cannot be negative")
        vol_dim = (
            len(self.volatility) if isinstance(self.volatility, (list, tuple)) else 1
        )
        div_dim = (
            len(self.dividend_yield)
            if isinstance(self.dividend_yield, (list, tuple))
            else 1
        )
        if vol_dim != self.dimension or div_dim != self.dimension:
            raise ValueError(
                "Volatility and dividend yield must match the dimension of the spot price"
            )

    def set_payoff_function(self, payoff_func):
        """
        Set the payoff function for the option.

        Parameters:
        ----------
        payoff_func : callable
            A function that takes asset prices and returns the option payoff.
        """
        if not callable(payoff_func):
            raise ValueError("payoff_func must be a callable function")
        self.payoff_func = payoff_func

    def bjerksund_stensland(self, option_type="call"):
        """
        Price an American option using the Bjerksund-Stensland model.

        Parameters:
        ----------
        option_type : str
            'call' or 'put'

        Returns:
        -------
        float
            The price of the American option.
        """
        if self.dimension != 1:
            return None
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        start = time.time()
        model = BjerksundStensland(
            self.spot_price,
            self.strike,
            self.time_to_maturity,
            self.risk_free_rate,
            self.risk_free_rate - self.dividend_yield,
            self.volatility,
            option_type=option_type.lower(),
        )
        self.bjs_time = time.time() - start
        price = model.calculate_price()
        self.bjs_price = price

        return price

    def stochastic_mesh(self, num_paths=100):
        """
        Price an American option using the stochastic mesh method.

        Parameters:
        ----------
        num_paths : int
            Number of simulated paths for the underlying asset

        Returns:
        -------
        float
            The price of the American option.
        """
        start = time.time()
        low_price, high_price, smm_mesh_dict, smm_path_dict = price_option(
            np.transpose(self.asset_paths, (2, 0, 1)),
            self.payoff_func,
            self.spot_price,
            self.risk_free_rate - self.dividend_yield,
            self.volatility,
            self.risk_free_rate,
            self.dividend_yield,
            self.time_to_maturity,
            self.time_to_maturity / len(self.execution_times),
            num_paths,
        )
        self.smm_time = time.time() - start
        self.smm_mesh_price, self.smm_path_price = high_price, low_price
        self.smm_mesh_dict, self.smm_path_dict = smm_mesh_dict, smm_path_dict

        return low_price, high_price

    def longstaff_schwartz(self, basis_fn=None, num_paths=1000):
        """
        Price an American option using the Longstaff-Schwartz method.

        Parameters:
        ----------
        option_type : str
            'call' or 'put'
        num_paths : int
            Number of simulated paths for the underlying asset

        Returns:
        -------
        float
            The price of the American option.
        """
        start = time.time()
        price, matrices_dict = lsm_price_multi(
            self.asset_paths,
            self.spot_price,
            self.risk_free_rate,
            self.volatility,
            self.corr,
            self.time_to_maturity,
            len(self.execution_times),
            num_paths,
            self.payoff_func,
            basis_fn,
            self.seed,
            True,
        )
        self.lsm_time = time.time() - start
        self.lsm_price = price
        self.lsm_dict = matrices_dict

        return price

    def full_data(self):
        return {
            "smm_mesh_price": self.smm_mesh_price,
            "smm_path_price": self.smm_path_price,
            "lsm_price": self.lsm_price,
            "bjs_price": self.bjs_price,
            "smm_time": self.smm_time,
            "lsm_time": self.lsm_time,
            "bjs_time": self.bjs_time,
        }
