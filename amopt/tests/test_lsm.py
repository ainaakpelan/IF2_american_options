# tests/test_lsm.py
import math
import numpy as np
import pytest

import lsm  # <- twój moduł z notebooka

# --------------------
#  Black–Scholes helpers
# --------------------
def bs_eur_call(S, K, r, sigma, T):
    """Black–Scholes European call."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)

def bs_eur_put(S, K, r, sigma, T):
    """European put via put–call parity."""
    C = bs_eur_call(S, K, r, sigma, T)
    return C + K * math.exp(-r * T) - S

# --------------------
#  global parameters (can be overridden per-case)
# --------------------
r       = 0.06
T       = 1.0
M       = 40
SEED    = 42
EARLY_EX_PREMIUM_CAP = 0.40        # 40 % ceny europejskiej

# =========================================================
#  1) American CALL (bez dywidendy) == European CALL
# =========================================================
@pytest.mark.parametrize("S0,K,sigma,Npaths", [
    (36, 36, 0.20, 60_000),
    (80, 75, 0.15, 80_000),
    (50, 60, 0.30, 80_000),
])
def test_call_equals_european(S0, K, sigma, Npaths):
    am_call = lsm.lsm_price_multi(
        S0=[S0], r=r, sigma=[sigma], corr=[[1]], T=T,
        M=M, N=Npaths, payoff_fn=lsm.payoff_call(K), seed=SEED,
    )
    eu_call = bs_eur_call(S0, K, r, sigma, T)
    # bez dywidendy nie opłaca się ćwiczyć calla
    assert math.isclose(am_call, eu_call, rel_tol=0.05)

# =========================================================
#  2) American PUT >= European PUT  (dynamiczny próg)
# =========================================================
@pytest.mark.parametrize("S0,K,sigma,Npaths", [
    (36, 40, 0.20, 60_000),
    (50, 60, 0.15, 100_000),
    (100,110, 0.30, 50_000),
])
def test_put_above_european(S0, K, sigma, Npaths):
    am_put = lsm.lsm_price_multi(
        S0=[S0], r=r, sigma=[sigma], corr=[[1]], T=T,
        M=M, N=Npaths, payoff_fn=lsm.payoff_put(K), seed=SEED,
    )
    eu_put = bs_eur_put(S0, K, r, sigma, T)

    # 1) arbitraż
    assert am_put >= eu_put

    # 2) sensowny rozmiar premii (<= 40 % europejskiej)
    assert am_put - eu_put <= EARLY_EX_PREMIUM_CAP * eu_put + 1e-9

# =========================================================
#  3) Monotoniczność po strajku (CALL ↓, PUT ↑)
# =========================================================
@pytest.mark.parametrize("K_low,K_high", [(30, 40), (35, 45)])
def test_strike_monotonicity_call(K_low, K_high):
    low = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T, M, 60_000,
        payoff_fn=lsm.payoff_call(K_low), seed=SEED
    )
    high = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T, M, 60_000,
        payoff_fn=lsm.payoff_call(K_high), seed=SEED
    )
    assert low > high

@pytest.mark.parametrize("K_low,K_high", [(30, 40), (35, 45)])
def test_strike_monotonicity_put(K_low, K_high):
    low = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T, M, 60_000,
        payoff_fn=lsm.payoff_put(K_low), seed=SEED
    )
    high = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T, M, 60_000,
        payoff_fn=lsm.payoff_put(K_high), seed=SEED
    )
    assert low < high

# =========================================================
# 4) Powtarzalność przy tym samym seedzie
# =========================================================
def test_reproducibility():
    price1 = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T,
        M, 50_000, payoff_fn=lsm.payoff_put(40), seed=123
    )
    price2 = lsm.lsm_price_multi(
        [36], r, [0.20], [[1]], T,
        M, 50_000, payoff_fn=lsm.payoff_put(40), seed=123
    )
    assert price1 == price2

# =========================================================
# 5) Zero-payoff sanity (call_binary z wysoką barierą)
# =========================================================
def test_zero_payoff():
    price = lsm.lsm_price_multi(
        S0=[50, 30], r=r, sigma=[0.20, 0.20], corr=[[1,0],[0,1]], T=T,
        M=M, N=60_000,
        payoff_fn=lsm.payoff_call_binary(K=55, idx_pay=0, idx_cond=1, H=1_000),
        seed=SEED,
    )
    assert price < 1e-3
