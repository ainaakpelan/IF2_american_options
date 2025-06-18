import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from amopt.pricing.pricing import AmericanOptionPricer
from amopt.models.longstaff_schwartz import payoff_call, payoff_put

def generate_comparison_plot(
    spot_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    execution_steps: int,
    n_paths_lsm: int,
    n_paths_mesh: int,
    strike_min: float,
    strike_max: float,
    strike_steps: int,
    option_type: str = "call",
) -> str:
    """
    Generuje wykres porównawczy cen opcji typu call/put dla trzech metod
    w zależności od strike price. Zwraca base64 PNG.
    """
    M = execution_steps
    execution_times = np.linspace(0, time_to_maturity, M + 1)
    strike_range = np.linspace(strike_min, strike_max, strike_steps)

    prices_lsm = []
    prices_mesh = []
    prices_bjs = []

    for K in strike_range:
        pricer = AmericanOptionPricer(
            [spot_price],
            K,
            risk_free_rate,
            [volatility],
            time_to_maturity,
            execution_times,
            n_paths_mesh,
            corr=[[1]],
            dividend_yield=[dividend_yield],
            seed=42
        )

        # Longstaff-Schwartz
        if option_type == "call":
            pricer.set_payoff_function(payoff_call(K))
        else:
            pricer.set_payoff_function(payoff_put(K))
        prices_lsm.append(pricer.longstaff_schwartz(num_paths=n_paths_lsm))

        # Stochastic Mesh
        if option_type == "call":
            pricer.set_payoff_function(lambda S: np.maximum(S - K, 0))
        else:
            pricer.set_payoff_function(lambda S: np.maximum(K - S, 0))
        prices_mesh.append(pricer.stochastic_mesh(n_paths_mesh))

        # Bjerksund-Stensland
        prices_bjs.append(pricer.bjerksund_stensland(option_type=option_type))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(strike_range, prices_lsm, label='Longstaff-Schwartz (LSM)')
    ax.plot(strike_range, prices_mesh, label='Stochastic Mesh')
    ax.plot(strike_range, prices_bjs, label='Bjerksund-Stensland')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Option Price')
    ax.set_title(f'American {option_type.title()} Option Price vs Strike')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return data
