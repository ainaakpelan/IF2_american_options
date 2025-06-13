import numpy as np
from typing import Callable, Optional, Tuple

from .american_option_pricer import AmericanOptionPricer
from .data_structures import ModelParams
from .simulation import simulate_gbm_paths
from .stochastic_mesh import mesh_estimator, path_estimator

class StochasticMeshPricer(AmericanOptionPricer):
    """
    Andersen–Broadie stochastic mesh pricer optimized.

    Parametry:
    - model_params: dane modelu GBM.
    - payoff: funkcja wypłaty na pełnych ścieżkach (N, M+1, d) -> (N, M+1).
    - mesh_payoff_fn: funkcja wypłaty na mesh (d, N, M+1) -> (N, M+1), opcjonalna.
    - antithetic: czy używać wariatów antytetycznych.
    - seed: ziarno RNG.
    - n_paths_mesh: liczba ścieżek używana do górnego i dolnego oszacowania (jeśli None, bierze model_params.n_paths).

    Metoda price():
    - Jeśli przekażesz `paths` (kopia mesh), użyje ich jako siatki.
    - W przeciwnym razie symuluje `n_paths_mesh` ścieżek GBM.
    - Zwraca (low_price, high_price).

    Atrybuty po wywołaniu price():
    - mesh_option_prices: macierz cen opcji (upper estimates).
    """
    def __init__(
        self,
        model_params: ModelParams,
        payoff: Callable[[np.ndarray], np.ndarray],
        mesh_payoff_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        antithetic: bool = False,
        seed: Optional[int] = None,
        n_paths_mesh: Optional[int] = None
    ):
        super().__init__(model_params, payoff)
        self.mesh_payoff_fn = mesh_payoff_fn or (lambda X: self.payoff(np.transpose(X, (1,2,0))))
        self.antithetic = antithetic
        self.seed = seed
        # liczba ścieżek dla mesh i dolnego oszacowania
        self.n_paths_mesh = n_paths_mesh if n_paths_mesh is not None else self.n_paths
        self.mesh_option_prices: Optional[np.ndarray] = None

    def price(
        self,
        paths: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Wycena opcji amerykańskiej metodą stochastic mesh.

        Parametry:
        - paths: opcjonalne ścieżki (N, M+1, d) do użycia jako mesh.
        Zwraca:
        - low_price: dolna granica ceny (path estimator),
        - high_price: górna granica ceny (mesh estimator).
        """
        # 1) przygotowanie mesh
        if paths is None:
            paths = simulate_gbm_paths(
                S0         = self.S0,
                r          = self.r,
                sigma      = self.sigma,
                q          = self.dividend_yield,
                corr       = self.corr,
                T          = self.T,
                M          = self.n_steps,
                N          = self.n_paths_mesh,
                seed       = self.seed,
                antithetic = self.antithetic
            )
        mesh = np.transpose(paths, (2, 0, 1))  # (d, b, m)
        dt   = self.T / self.n_steps

        # 2) górne oszacowanie (mesh estimator)
        option_prices = mesh_estimator(
            asset_price_mesh = mesh,
            payoff_func      = self.mesh_payoff_fn,
            vol              = self.sigma,
            rf_rate          = self.r,
            dividend_yield   = self.dividend_yield,
            dt               = dt
        )
        self.mesh_option_prices = option_prices
        high_price = float(option_prices[0, 0])

        # 3) dolne oszacowanie (path estimator wektorowo)
        # drift musi być tablicą kształtu (d,)
        drift_array = np.full_like(self.S0, self.r)
        low_price = path_estimator(
            asset_price_mesh = mesh,
            payoff_func      = self.mesh_payoff_fn,
            spot_prices      = self.S0,
            drift            = drift_array,
            vol              = self.sigma,
            rf_rate          = self.r,
            dividend_yield   = self.dividend_yield,
            max_time         = self.T,
            dt               = dt,
            n_paths          = self.n_paths_mesh
        )

        return low_price, high_price
