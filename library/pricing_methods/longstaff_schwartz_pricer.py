import numpy as np
from typing import Callable, Optional
from .american_option_pricer import AmericanOptionPricer
from .data_structures import ModelParams
from .simulation import simulate_gbm_paths
# import builders z lsm.py
from polynom import (
    _build_basis,
    _build_basis_with_cross,
    _build_basis_laguerre,
    _build_basis_laguerre_multid,
    _build_basis_weighted_laguerre
)

class LongstaffSchwartzPricer(AmericanOptionPricer):
    """
    Longstaff–Schwartz pricer z wstrzykiwalną funkcją bazową:
      - basis_fn: Callable[[np.ndarray], np.ndarray] 
        przyjmuje S_itm (kropki ćwiczeń): shape (n,) lub (n,d),
        zwraca macierz regresji X o wymiarze (n, K).
    Jeśli basis_fn=None, używany jest domyślny monomial stopnia `degree`.
    """
    def __init__(
        self,
        model_params: ModelParams,
        payoff: Callable[[np.ndarray], np.ndarray],
        degree: int = 2,
        basis_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        antithetic: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(model_params, payoff)
        self.degree     = degree
        self.basis_fn   = basis_fn or (lambda S: _build_basis(
            S.reshape(-1,1) if S.ndim==1 else S,
            degree=self.degree
        ))
        self.antithetic = antithetic
        self.seed       = seed

    def _design_matrix(self, S_itm: np.ndarray) -> np.ndarray:
        """
        Wywołuje basis_fn. S_itm może być 1D (shape (n,)) lub 2D (n,d).
        """
        return self.basis_fn(S_itm)

    def price(self, paths: Optional[np.ndarray] = None) -> float:
        # 1) symulacja ścieżek (N, M+1, d)
        if paths is None:
            paths = simulate_gbm_paths(
                S0=self.S0,
                r=self.r,
                sigma=self.sigma,
                q=self.dividend_yield,
                corr=self.corr,
                T=self.T,
                M=self.n_steps,
                N=self.n_paths,
                seed=self.seed,
                antithetic=self.antithetic,
            )

        N, M_plus_one, d = paths.shape
        dt = self.T / self.n_steps

        # 2) pay-off w T
        cashflow = self.payoff(paths[:, -1, :])
        exercise_time = np.full(N, M_plus_one - 1, dtype=int)

        # 3) backward induction
        for t in range(M_plus_one - 2, 0, -1):
            # bieżąca cena: jednowymiarowa lub wielowymiarowa
            St = paths[:, t, 0] if d == 1 else paths[:, t, :]
            intrinsic = self.payoff(paths[:, t, :])
            itm = intrinsic > 0
            if not itm.any():
                continue

            # regresja continuation value
            S_itm = St[itm]               # (n_itm,) lub (n_itm,d)
            Y     = cashflow[itm] * np.exp(-self.r * dt)
            A     = self._design_matrix(S_itm)
            coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)
            cont = A.dot(coeffs)

            # decyzja/exercise
            ex = intrinsic[itm] > cont
            idx = np.where(itm)[0][ex]
            cashflow[idx]      = intrinsic[itm][ex]
            exercise_time[idx] = t

        # 4) diskontowanie do 0 i średnia
        discounts = np.exp(-self.r * dt * exercise_time)
        return float(np.mean(cashflow * discounts))
