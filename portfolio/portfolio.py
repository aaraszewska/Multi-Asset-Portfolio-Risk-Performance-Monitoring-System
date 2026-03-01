"""Portfolio class that tracks positions and aggregates risk/performance metrics."""

from __future__ import annotations

import pandas as pd

from portfolio import metrics as m


class Portfolio:
    """Buy-and-hold multi-asset portfolio.

    Parameters
    ----------
    prices:
        DataFrame with asset prices indexed by date.  Columns are asset
        tickers (e.g. ``["BTC", "ETH", "GOLD", "NASDAQ"]``).
    weights:
        Target allocation weights as a mapping ``{ticker: weight}``.
        Weights must sum to 1.  If omitted, equal weights are used.
    initial_nav:
        Starting portfolio value in base currency (default 1 000 000).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: dict[str, float] | None = None,
        initial_nav: float = 1_000_000.0,
    ) -> None:
        self.prices = prices
        self.initial_nav = initial_nav

        if weights is None:
            n = len(prices.columns)
            weights = {col: 1.0 / n for col in prices.columns}

        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1, got {total}.")

        self.weights: dict[str, float] = weights

        # Units are fixed at inception (buy-and-hold)
        initial_prices = prices.iloc[0]
        self.units = pd.Series(
            {
                asset: (initial_nav * weights[asset]) / initial_prices[asset]
                for asset in prices.columns
            }
        )

    # ------------------------------------------------------------------
    # NAV and PnL
    # ------------------------------------------------------------------

    def nav(self) -> pd.Series:
        """Portfolio Net Asset Value over time."""
        return m.compute_nav(self.prices, self.units)

    def pnl(self) -> pd.Series:
        """Daily Profit & Loss."""
        return m.compute_pnl(self.nav())

    def cumulative_pnl(self) -> pd.Series:
        """Cumulative PnL (NAV minus initial NAV)."""
        return self.nav() - self.nav().iloc[0]

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def returns(self) -> pd.Series:
        """Daily log returns of the portfolio NAV."""
        return m.compute_returns(self.nav())

    def cumulative_returns(self) -> pd.Series:
        """Cumulative simple returns of the portfolio NAV."""
        return m.compute_cumulative_returns(self.nav())

    def asset_returns(self) -> pd.DataFrame:
        """Daily log returns per asset."""
        return m.compute_returns(self.prices)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def volatility(self) -> float:
        """Annualised portfolio volatility."""
        return float(m.compute_volatility(self.returns()))

    def asset_volatilities(self) -> pd.Series:
        """Per-asset annualised volatilities."""
        return m.compute_volatility(self.asset_returns())

    def max_drawdown(self) -> float:
        """Maximum drawdown of portfolio NAV."""
        return m.compute_max_drawdown(self.nav())

    def asset_max_drawdowns(self) -> pd.Series:
        """Maximum drawdown for each individual asset."""
        return self.prices.apply(m.compute_max_drawdown)

    def var(self, confidence: float = 0.95, method: str = "historical") -> float:
        """Value at Risk at the given confidence level."""
        return m.compute_var(self.returns(), confidence=confidence, method=method)

    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation matrix of asset returns."""
        return m.compute_correlation(self.asset_returns())

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------

    def exposure(self) -> pd.Series:
        """Average asset exposure as a fraction of NAV over the simulation."""
        nav_series = self.nav()
        return pd.Series(
            {
                asset: float((self.prices[asset] * self.units[asset] / nav_series).mean())
                for asset in self.prices.columns
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of key performance and risk indicators."""
        nav_series = self.nav()
        cum_return_pct = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100

        return {
            "initial_nav": float(nav_series.iloc[0]),
            "final_nav": float(nav_series.iloc[-1]),
            "cumulative_return_pct": float(cum_return_pct),
            "annualized_volatility": self.volatility(),
            "max_drawdown": self.max_drawdown(),
            "var_95_historical": self.var(0.95, "historical"),
            "var_95_parametric": self.var(0.95, "parametric"),
            "asset_volatilities": self.asset_volatilities().to_dict(),
            "asset_max_drawdowns": self.asset_max_drawdowns().to_dict(),
            "exposure": self.exposure().to_dict(),
            "correlation_matrix": self.correlation_matrix().to_dict(),
        }
