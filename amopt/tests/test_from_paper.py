"""
Porównanie implementacji Longstaff-Schwartz z wartościami z Longstaff & Schwartz (2001),
Sekcja 3, Tabela 1.

Wymaga:
    - lsm.py
    - pytest
"""
import math
import pytest
import lsm

# --- parametry wspólne ----------------------------------
RISK_FREE = 0.06
STRIKE     = 40          # w tabeli autorzy używali K = 40
M_STEPS    = 50          # 50 exercise points per year
N_PATHS    = 60_000      # rozsądny kompromis dokładność/czas
SEED       = 123

# --- wiersze wyjęte z tabeli 1 (American values) --------
REFERENCE = [
    # (S0 , sigma , T , american_value)
    (36 , 0.20 , 1 , 4.478),
    (36 , 0.20 , 2 , 4.840),
    (36 , 0.40 , 1 , 7.101),
    (36 , 0.40 , 2 , 8.508),
    (38 , 0.20 , 1 , 3.250),
    (38 , 0.20 , 2 , 3.745),
    (38 , 0.40 , 1 , 6.148),
    (38 , 0.40 , 2 , 7.670),
    (40 , 0.20 , 1 , 2.314),
    (40 , 0.20 , 2 , 2.885),
    (40 , 0.40 , 1 , 5.312),
    (40 , 0.40 , 2 , 6.920),
]

@pytest.mark.parametrize("S0,sigma,T,ref_val", REFERENCE)
def test_lsm_matches_paper(S0, sigma, T, ref_val):
    """
    Sprawdzamy, czy nasza implementacja Longstaff–Schwartz
    daje wynik zgodny z wartością z pracy (<5 % względnego odchylenia).
    """
    price = lsm.lsm_price_multi(
        S0=[S0],
        r=RISK_FREE,
        sigma=[sigma],
        corr=[[1]],           # 1-wymiarowy GBM
        T=T,
        M=int(M_STEPS * T),   # tyle kroków, ile lat * 50
        N=N_PATHS,
        payoff_fn=lsm.payoff_put(STRIKE),
        seed=SEED,
    )

    rel_err = abs(price - ref_val) / ref_val
    assert rel_err < 0.05, (
        f"W przypadku S0={S0}, σ={sigma}, T={T} "
        f"otrzymano {price:.3f} vs {ref_val:.3f} (|%err|={rel_err:.1%})"
    )
