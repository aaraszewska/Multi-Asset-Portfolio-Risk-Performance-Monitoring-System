"""Risk and performance metric calculations for a multi-asset portfolio."""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def compute_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily log returns.

    Parameters
    ----------
    prices:
        Price series or DataFrame of price series.

    Returns
    -------
    Series or DataFrame of log returns (first row is NaN and is dropped).
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_cumulative_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute cumulative simple returns from price series.

    The cumulative return at day t is  (P(t) / P(0)) - 1.
    """
    return prices / prices.iloc[0] - 1


def compute_nav(prices: pd.DataFrame, units: pd.Series) -> pd.Series:
    """Compute portfolio Net Asset Value (NAV) at each date.

    Parameters
    ----------
    prices:
        DataFrame with asset prices indexed by date.
    units:
        Number of units held per asset (constant buy-and-hold).

    Returns
    -------
    pd.Series
        NAV indexed by date.
    """
    return (prices * units).sum(axis=1)


def compute_pnl(nav: pd.Series) -> pd.Series:
    """Compute daily profit and loss from NAV series."""
    return nav.diff().dropna()


def compute_volatility(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
) -> float | pd.Series:
    """Compute standard-deviation-based volatility.

    Parameters
    ----------
    returns:
        Daily log-return series or DataFrame.
    annualize:
        If ``True`` (default), multiply by sqrt(252) to annualise.

    Returns
    -------
    Scalar float for a Series input, pd.Series for a DataFrame input.
    """
    vol = returns.std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute the maximum drawdown of a price (or NAV) series.

    Maximum drawdown = min( (P(t) - peak(t)) / peak(t) )  for all t.

    Returns
    -------
    float
        A non-positive value representing the worst peak-to-trough decline.
    """
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute Value at Risk (VaR) for a return series.

    Parameters
    ----------
    returns:
        Daily log-return series.
    confidence:
        Confidence level (e.g. 0.95 for 95 % VaR).
    method:
        ``"historical"`` – empirical quantile of the return distribution.
        ``"parametric"`` – normal-distribution approximation.

    Returns
    -------
    float
        VaR expressed as a return (negative value indicates a loss).
    """
    if method == "historical":
        return float(returns.quantile(1 - confidence))
    if method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        return float(stats.norm.ppf(1 - confidence, loc=mu, scale=sigma))
    raise ValueError(f"Unknown VaR method: '{method}'. Use 'historical' or 'parametric'.")


def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix of asset returns.

    Parameters
    ----------
    returns:
        DataFrame where each column is a daily return series for one asset.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    return returns.corr()
