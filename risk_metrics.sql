-- ==========================================
-- RISK METRICS
-- ==========================================

-- Portfolio Returns
CREATE OR ALTER VIEW v_portfolio_returns AS
SELECT
    [date],
    nav,
    pnl_daily,
    pnl_daily / NULLIF(LAG(nav) OVER (ORDER BY [date]),0) AS portfolio_return
FROM v_portfolio_daily;


-- Rolling 30D Volatility
CREATE OR ALTER VIEW v_portfolio_vol30 AS
WITH stats AS (
    SELECT
        [date],
        portfolio_return,
        AVG(portfolio_return) OVER (
            ORDER BY [date]
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS mean_30,
        AVG(portfolio_return * portfolio_return) OVER (
            ORDER BY [date]
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS mean_sq_30
    FROM v_portfolio_returns
    WHERE portfolio_return IS NOT NULL
)
SELECT
    [date],
    portfolio_return,
    SQRT(ABS(mean_sq_30 - (mean_30 * mean_30))) AS vol_30
FROM stats;


-- Max Drawdown
CREATE OR ALTER VIEW v_portfolio_drawdown AS
WITH nav_series AS (
    SELECT
        [date],
        nav,
        MAX(nav) OVER (
            ORDER BY [date]
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS rolling_peak
    FROM v_portfolio_daily
)
SELECT
    [date],
    nav,
    rolling_peak,
    (nav - rolling_peak) / NULLIF(rolling_peak,0) AS drawdown_pct
FROM nav_series;


-- Sharpe Summary
CREATE OR ALTER VIEW v_portfolio_sharpe_summary AS
WITH stats AS (
    SELECT
        AVG(portfolio_return) AS mean_daily,
        STDEV(portfolio_return) AS stdev_daily
    FROM v_portfolio_returns
    WHERE portfolio_return IS NOT NULL
)
SELECT
    mean_daily,
    stdev_daily,
    mean_daily * 252 AS annual_return,
    stdev_daily * SQRT(252.0) AS annual_vol,
    (mean_daily * 252) / NULLIF(stdev_daily * SQRT(252.0),0) AS sharpe_ratio
FROM stats;
