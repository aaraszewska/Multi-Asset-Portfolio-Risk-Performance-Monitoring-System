"""Tests for portfolio/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from portfolio import metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_prices() -> pd.Series:
    """A simple monotonically-increasing price series."""
    return pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])


@pytest.fixture
def flat_prices() -> pd.Series:
    """A price series that never moves (zero volatility, no drawdown)."""
    return pd.Series([100.0] * 10)


@pytest.fixture
def multi_prices() -> pd.DataFrame:
    """A small multi-asset price DataFrame."""
    rng = np.random.default_rng(0)
    n = 50
    data = {
        "A": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))),
        "B": 200 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_length(self, simple_prices):
        result = metrics.compute_returns(simple_prices)
        assert len(result) == len(simple_prices) - 1

    def test_values_approx(self, simple_prices):
        result = metrics.compute_returns(simple_prices)
        expected_first = np.log(110.0 / 100.0)
        assert result.iloc[0] == pytest.approx(expected_first)

    def test_flat_prices_zero_returns(self, flat_prices):
        result = metrics.compute_returns(flat_prices)
        assert (result == 0).all()

    def test_dataframe_input(self, multi_prices):
        result = metrics.compute_returns(multi_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(multi_prices) - 1, multi_prices.shape[1])


# ---------------------------------------------------------------------------
# compute_cumulative_returns
# ---------------------------------------------------------------------------

class TestComputeCumulativeReturns:
    def test_first_value_is_zero(self, simple_prices):
        result = metrics.compute_cumulative_returns(simple_prices)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_monotone_increasing(self, simple_prices):
        result = metrics.compute_cumulative_returns(simple_prices)
        assert result.is_monotonic_increasing

    def test_final_value(self, simple_prices):
        result = metrics.compute_cumulative_returns(simple_prices)
        expected = simple_prices.iloc[-1] / simple_prices.iloc[0] - 1
        assert result.iloc[-1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_nav
# ---------------------------------------------------------------------------

class TestComputeNav:
    def test_constant_units(self, multi_prices):
        units = pd.Series({"A": 2.0, "B": 3.0})
        nav = metrics.compute_nav(multi_prices, units)
        expected_first = 2.0 * multi_prices["A"].iloc[0] + 3.0 * multi_prices["B"].iloc[0]
        assert nav.iloc[0] == pytest.approx(expected_first)

    def test_length(self, multi_prices):
        units = pd.Series({"A": 1.0, "B": 1.0})
        nav = metrics.compute_nav(multi_prices, units)
        assert len(nav) == len(multi_prices)


# ---------------------------------------------------------------------------
# compute_pnl
# ---------------------------------------------------------------------------

class TestComputePnl:
    def test_length(self):
        nav = pd.Series([100.0, 105.0, 103.0, 110.0])
        pnl = metrics.compute_pnl(nav)
        assert len(pnl) == len(nav) - 1

    def test_values(self):
        nav = pd.Series([100.0, 105.0, 103.0])
        pnl = metrics.compute_pnl(nav)
        assert pnl.iloc[0] == pytest.approx(5.0)
        assert pnl.iloc[1] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# compute_volatility
# ---------------------------------------------------------------------------

class TestComputeVolatility:
    def test_flat_prices_zero_vol(self, flat_prices):
        returns = metrics.compute_returns(flat_prices)
        assert metrics.compute_volatility(returns) == pytest.approx(0.0)

    def test_annualisation(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        daily_std = returns["A"].std()
        annualised = metrics.compute_volatility(returns["A"])
        assert annualised == pytest.approx(daily_std * np.sqrt(252))

    def test_no_annualisation(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        daily_std = returns["A"].std()
        vol = metrics.compute_volatility(returns["A"], annualize=False)
        assert vol == pytest.approx(daily_std)

    def test_dataframe_returns_series(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        vol = metrics.compute_volatility(returns)
        assert isinstance(vol, pd.Series)
        assert set(vol.index) == {"A", "B"}


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    def test_monotone_increasing_no_drawdown(self, simple_prices):
        dd = metrics.compute_max_drawdown(simple_prices)
        assert dd == pytest.approx(0.0)

    def test_known_drawdown(self):
        prices = pd.Series([100.0, 90.0, 80.0, 95.0, 105.0])
        dd = metrics.compute_max_drawdown(prices)
        # Peak = 100, trough = 80 → drawdown = (80-100)/100 = -0.20
        assert dd == pytest.approx(-0.20)

    def test_returns_non_positive(self, multi_prices):
        dd = metrics.compute_max_drawdown(multi_prices["A"])
        assert dd <= 0.0


# ---------------------------------------------------------------------------
# compute_var
# ---------------------------------------------------------------------------

class TestComputeVar:
    @pytest.fixture
    def returns(self, multi_prices):
        return metrics.compute_returns(multi_prices["A"])

    def test_historical_var_negative(self, returns):
        var = metrics.compute_var(returns, confidence=0.95, method="historical")
        assert var < 0

    def test_parametric_var_negative(self, returns):
        var = metrics.compute_var(returns, confidence=0.95, method="parametric")
        assert var < 0

    def test_historical_vs_parametric_similar(self, returns):
        h_var = metrics.compute_var(returns, confidence=0.95, method="historical")
        p_var = metrics.compute_var(returns, confidence=0.95, method="parametric")
        # Both should be in a reasonable neighbourhood
        assert abs(h_var - p_var) < 0.05

    def test_higher_confidence_gives_worse_var(self, returns):
        var_95 = metrics.compute_var(returns, confidence=0.95)
        var_99 = metrics.compute_var(returns, confidence=0.99)
        assert var_99 <= var_95

    def test_invalid_method_raises(self, returns):
        with pytest.raises(ValueError, match="Unknown VaR method"):
            metrics.compute_var(returns, method="montecarlo")


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------

class TestComputeCorrelation:
    def test_shape(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        corr = metrics.compute_correlation(returns)
        assert corr.shape == (2, 2)

    def test_diagonal_is_one(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        corr = metrics.compute_correlation(returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0)

    def test_symmetric(self, multi_prices):
        returns = metrics.compute_returns(multi_prices)
        corr = metrics.compute_correlation(returns)
        np.testing.assert_allclose(corr.values, corr.values.T)
