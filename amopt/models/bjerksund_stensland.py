import numpy as np
from scipy.stats import norm

class BjerksundStensland:
    """
    Bjerksund-Stensland (1993) model for American option pricing.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    b : float
        Cost of carry (r - q)
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    """

    def __init__(self, S, K, T, r, b, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.b = b
        self.sigma = sigma
        self.option_type = option_type.lower()

        if T <= 0:
            raise ValueError("Time to maturity must be positive")
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

    def _beta(self):
        term = ((self.b / self.sigma**2 - 0.5)**2 + 
                2 * self.r / self.sigma**2)
        return (0.5 - self.b / self.sigma**2) + np.sqrt(term)

    def _X(self, beta):
        B_inf = beta / (beta - 1) * self.K
        if self.r != self.b:
            B_0 = max(self.K, self.r / (self.r - self.b) * self.K)
        else:
            B_0 = float('inf')

        h = -(self.b * self.T + 2 * self.sigma * np.sqrt(self.T)) * (B_0 / (B_inf - B_0))
        return B_0 + (B_inf - B_0) * (1 - np.exp(h))

    def _phi(self, S, gamma, H, I):
        lambda_ = -self.r + gamma * self.b + 0.5 * gamma * (gamma - 1) * self.sigma**2
        kappa = 2 * self.b / self.sigma**2 + 2 * gamma - 1
        sigma_sqrt_T = self.sigma * np.sqrt(self.T)

        d1 = -(np.log(S / H) + (self.b + (gamma - 0.5) * self.sigma**2) * self.T) / sigma_sqrt_T
        d2 = -(np.log(S / H) + 2 * np.log(I / S) + (self.b + (gamma - 0.5) * self.sigma**2) * self.T) / sigma_sqrt_T

        term1 = np.exp(lambda_ * self.T) * S**gamma * norm.cdf(d1)
        term2 = np.exp(lambda_ * self.T) * I**kappa * S**(gamma - kappa) * norm.cdf(d2)

        return term1 - term2

    def calculate_call_price(self):
        beta = self._beta()
        X = self._X(beta)
        alpha = (X - self.K) * X**(-beta)

        value = (
            alpha * self.S**beta
            - alpha * self._phi(self.S, beta, X, X)
            + self._phi(self.S, 1.0, X, X)
            - self._phi(self.S, 1.0, self.K, X)
            - self.K * self._phi(self.S, 0.0, X, X)
            + self.K * self._phi(self.S, 0.0, self.K, X)
        )
        return value

    def calculate_put_price(self):
        transformed = BjerksundStensland(
            S=self.K,
            K=self.S,
            T=self.T,
            r=self.r - self.b,
            b=-self.b,
            sigma=self.sigma,
            option_type='call'  # transform to call
        )
        return transformed.calculate_call_price()

    def calculate_price(self):
        if self.option_type == 'call':
            return self.calculate_call_price()
        else:
            return self.calculate_put_price()
