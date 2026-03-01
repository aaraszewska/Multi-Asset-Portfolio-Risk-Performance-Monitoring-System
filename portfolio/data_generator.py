"""Simulated daily price data generator using Geometric Brownian Motion (GBM)."""

import numpy as np
import pandas as pd

# Per-asset GBM parameters (daily drift and daily volatility)
ASSET_PARAMS = {
    "BTC": {"initial_price": 30_000.0, "drift": 0.0010, "volatility": 0.0400},
    "ETH": {"initial_price": 2_000.0, "drift": 0.0008, "volatility": 0.0450},
    "GOLD": {"initial_price": 1_900.0, "drift": 0.0002, "volatility": 0.0080},
    "NASDAQ": {"initial_price": 14_000.0, "drift": 0.0003, "volatility": 0.0120},
}


def generate_prices(
    n_days: int = 252,
    seed: int = 42,
    start_date: str = "2023-01-02",
) -> pd.DataFrame:
    """Return a DataFrame of simulated daily closing prices.

    Prices are generated via Geometric Brownian Motion:
        S(t) = S(t-1) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0, 1) and dt = 1 trading day.

    Parameters
    ----------
    n_days:
        Number of trading days to simulate.
    seed:
        Random-number-generator seed for reproducibility.
    start_date:
        First date in the returned index (business-day frequency).

    Returns
    -------
    pd.DataFrame
        Columns are asset tickers; index is a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    prices: dict[str, np.ndarray] = {}
    for asset, params in ASSET_PARAMS.items():
        s0 = params["initial_price"]
        mu = params["drift"]
        sigma = params["volatility"]

        # GBM log-returns for days 1 … n_days-1
        shocks = rng.standard_normal(n_days - 1)
        log_returns = (mu - 0.5 * sigma**2) + sigma * shocks

        series = np.empty(n_days)
        series[0] = s0
        series[1:] = s0 * np.exp(np.cumsum(log_returns))
        prices[asset] = series

    return pd.DataFrame(prices, index=dates)
