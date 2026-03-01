"""Tests for portfolio/portfolio.py and portfolio/data_generator.py."""

import numpy as np
import pandas as pd
import pytest

from portfolio.data_generator import generate_prices, ASSET_PARAMS
from portfolio.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Data-generator tests
# ---------------------------------------------------------------------------

class TestGeneratePrices:
    def test_shape(self):
        df = generate_prices(n_days=50, seed=0)
        assert df.shape == (50, len(ASSET_PARAMS))

    def test_columns_match_assets(self):
        df = generate_prices(n_days=10, seed=0)
        assert set(df.columns) == set(ASSET_PARAMS.keys())

    def test_first_row_matches_initial_prices(self):
        df = generate_prices(n_days=10, seed=0)
        for asset, params in ASSET_PARAMS.items():
            assert df[asset].iloc[0] == pytest.approx(params["initial_price"])

    def test_reproducible(self):
        df1 = generate_prices(n_days=30, seed=7)
        df2 = generate_prices(n_days=30, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_prices(n_days=30, seed=1)
        df2 = generate_prices(n_days=30, seed=2)
        assert not df1.equals(df2)

    def test_all_prices_positive(self):
        df = generate_prices(n_days=252, seed=42)
        assert (df > 0).all().all()

    def test_index_is_business_days(self):
        df = generate_prices(n_days=5, seed=0, start_date="2023-01-02")
        assert df.index.freq is None or isinstance(df.index, pd.DatetimeIndex)
        # No weekend days
        assert all(d.weekday() < 5 for d in df.index)


# ---------------------------------------------------------------------------
# Portfolio construction tests
# ---------------------------------------------------------------------------

class TestPortfolioConstruction:
    @pytest.fixture
    def prices(self):
        return generate_prices(n_days=30, seed=42)

    def test_equal_weights_default(self, prices):
        p = Portfolio(prices)
        n = len(prices.columns)
        for w in p.weights.values():
            assert w == pytest.approx(1.0 / n)

    def test_custom_weights(self, prices):
        weights = {"BTC": 0.4, "ETH": 0.3, "GOLD": 0.2, "NASDAQ": 0.1}
        p = Portfolio(prices, weights=weights)
        assert p.weights == weights

    def test_invalid_weights_raise(self, prices):
        bad_weights = {"BTC": 0.5, "ETH": 0.5, "GOLD": 0.5, "NASDAQ": 0.5}
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            Portfolio(prices, weights=bad_weights)

    def test_initial_nav_preserved(self, prices):
        initial_nav = 500_000.0
        p = Portfolio(prices, initial_nav=initial_nav)
        assert p.nav().iloc[0] == pytest.approx(initial_nav)


# ---------------------------------------------------------------------------
# NAV and PnL
# ---------------------------------------------------------------------------

class TestNavAndPnl:
    @pytest.fixture
    def portfolio(self):
        prices = generate_prices(n_days=100, seed=10)
        return Portfolio(prices)

    def test_nav_length(self, portfolio):
        assert len(portfolio.nav()) == 100

    def test_pnl_length(self, portfolio):
        assert len(portfolio.pnl()) == 99

    def test_cumulative_pnl_starts_at_zero(self, portfolio):
        assert portfolio.cumulative_pnl().iloc[0] == pytest.approx(0.0)

    def test_pnl_consistent_with_nav(self, portfolio):
        nav = portfolio.nav()
        pnl = portfolio.pnl()
        expected = nav.diff().dropna()
        pd.testing.assert_series_equal(pnl, expected)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

class TestReturns:
    @pytest.fixture
    def portfolio(self):
        prices = generate_prices(n_days=60, seed=99)
        return Portfolio(prices)

    def test_returns_length(self, portfolio):
        assert len(portfolio.returns()) == 59

    def test_cumulative_returns_starts_at_zero(self, portfolio):
        assert portfolio.cumulative_returns().iloc[0] == pytest.approx(0.0)

    def test_asset_returns_shape(self, portfolio):
        ar = portfolio.asset_returns()
        assert ar.shape == (59, 4)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    @pytest.fixture
    def portfolio(self):
        prices = generate_prices(n_days=252, seed=42)
        return Portfolio(prices)

    def test_volatility_positive(self, portfolio):
        assert portfolio.volatility() > 0

    def test_asset_volatilities_all_positive(self, portfolio):
        vols = portfolio.asset_volatilities()
        assert (vols > 0).all()

    def test_max_drawdown_non_positive(self, portfolio):
        assert portfolio.max_drawdown() <= 0

    def test_asset_max_drawdowns_all_non_positive(self, portfolio):
        dds = portfolio.asset_max_drawdowns()
        assert (dds <= 0).all()

    def test_var_negative(self, portfolio):
        assert portfolio.var(0.95, "historical") < 0
        assert portfolio.var(0.95, "parametric") < 0

    def test_correlation_matrix_shape(self, portfolio):
        corr = portfolio.correlation_matrix()
        assert corr.shape == (4, 4)

    def test_correlation_diagonal_ones(self, portfolio):
        corr = portfolio.correlation_matrix()
        np.testing.assert_allclose(np.diag(corr.values), 1.0)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

class TestExposure:
    @pytest.fixture
    def portfolio(self):
        prices = generate_prices(n_days=50, seed=3)
        return Portfolio(prices)

    def test_exposure_sums_to_one(self, portfolio):
        total = portfolio.exposure().sum()
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_assets_have_positive_exposure(self, portfolio):
        assert (portfolio.exposure() > 0).all()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    @pytest.fixture
    def summary(self):
        prices = generate_prices(n_days=252, seed=42)
        return Portfolio(prices).summary()

    def test_summary_keys(self, summary):
        expected_keys = {
            "initial_nav",
            "final_nav",
            "cumulative_return_pct",
            "annualized_volatility",
            "max_drawdown",
            "var_95_historical",
            "var_95_parametric",
            "asset_volatilities",
            "asset_max_drawdowns",
            "exposure",
            "correlation_matrix",
        }
        assert expected_keys <= set(summary.keys())

    def test_initial_nav_close_to_one_million(self, summary):
        assert summary["initial_nav"] == pytest.approx(1_000_000.0)

    def test_final_nav_positive(self, summary):
        assert summary["final_nav"] > 0
