"""Entry point: run a 1-year portfolio simulation and print a risk/performance report."""

from portfolio.data_generator import generate_prices
from portfolio.portfolio import Portfolio


def run_simulation(n_days: int = 252, seed: int = 42) -> Portfolio:
    """Simulate a multi-asset portfolio and print a report.

    Parameters
    ----------
    n_days:
        Number of trading days to simulate (default: 252 ≈ 1 year).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    Portfolio
        The configured Portfolio instance after simulation.
    """
    separator = "=" * 62

    print(separator)
    print("  Multi-Asset Portfolio Risk & Performance Monitoring System")
    print(separator)

    prices = generate_prices(n_days=n_days, seed=seed)
    assets = list(prices.columns)

    print(f"\nAssets       : {assets}")
    print(f"Trading days : {n_days}")
    print(f"Date range   : {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\nInitial prices:")
    for asset in assets:
        print(f"  {asset:<8}: {prices.iloc[0][asset]:>12,.2f}")

    portfolio = Portfolio(prices, initial_nav=1_000_000.0)
    s = portfolio.summary()

    print(f"\n{'-' * 40}")
    print("Portfolio Performance")
    print(f"{'-' * 40}")
    print(f"  Initial NAV          : ${s['initial_nav']:>14,.2f}")
    print(f"  Final NAV            : ${s['final_nav']:>14,.2f}")
    print(f"  Cumulative Return    : {s['cumulative_return_pct']:>13.2f} %")

    print(f"\n{'-' * 40}")
    print("Risk Metrics")
    print(f"{'-' * 40}")
    print(f"  Annualised Volatility: {s['annualized_volatility']:>13.4f}")
    print(f"  Max Drawdown         : {s['max_drawdown']:>13.4f}")
    print(f"  VaR 95% (historical) : {s['var_95_historical']:>13.4f}")
    print(f"  VaR 95% (parametric) : {s['var_95_parametric']:>13.4f}")

    print(f"\n{'-' * 40}")
    print("Asset Exposure (avg % of NAV)")
    print(f"{'-' * 40}")
    for asset, exp in s["exposure"].items():
        print(f"  {asset:<8}: {exp * 100:>6.2f} %")

    print(f"\n{'-' * 40}")
    print("Asset Annualised Volatilities")
    print(f"{'-' * 40}")
    for asset, vol in s["asset_volatilities"].items():
        print(f"  {asset:<8}: {vol:.4f}")

    print(f"\n{'-' * 40}")
    print("Asset Max Drawdowns")
    print(f"{'-' * 40}")
    for asset, dd in s["asset_max_drawdowns"].items():
        print(f"  {asset:<8}: {dd:.4f}")

    print(f"\n{'-' * 40}")
    print("Asset Return Correlation Matrix")
    print(f"{'-' * 40}")
    corr = portfolio.correlation_matrix().round(4)
    print(corr.to_string())

    print(f"\n{separator}\n")
    return portfolio


if __name__ == "__main__":
    run_simulation()
