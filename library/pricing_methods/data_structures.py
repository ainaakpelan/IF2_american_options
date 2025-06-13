
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ModelParams:
    """
    Container for market and simulation parameters required by
    AmericanOptionPricer and its subclasses.
    """
    spot: np.ndarray             # initial asset prices, shape (d,)
    rf_rate: float               # risk-free interest rate
    sigma: np.ndarray            # volatilities, shape (d,)
    corr: np.ndarray             # correlation matrix, shape (d, d)
    dividend_yield: np.ndarray   # continuous dividend yields, shape (d,)
    maturity: float              # time to expiry T
    n_steps: int                 # number of time steps M
    n_paths: int                 # number of Monte Carlo paths N

@dataclass
class OptionResult:
    """
    Unified container for path-wise results of an American option pricer.
    """
    cashflows: np.ndarray            # discounted cashflows, shape (n_paths, n_steps+1)
    execution_values: np.ndarray     # intrinsic values, shape (n_paths, n_steps+1)
    decision_matrix: np.ndarray      # exercise decisions (0 or 1), same shape
    exposures: Optional[np.ndarray] = None    # e.g. path-wise exposure, same shape
    exercise_count: Optional[int] = None      # total number of early exercises
