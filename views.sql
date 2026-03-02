-- ==========================================
-- CORE PORTFOLIO VIEWS
-- ==========================================

-- Daily positions
CREATE OR ALTER VIEW v_positions_daily AS
WITH trades_daily AS (
    SELECT
        CAST(trade_datetime AS date) AS [date],
        asset,
        SUM(CASE 
            WHEN side = 'BUY' THEN quantity
            WHEN side = 'SELL' THEN -quantity
        END) AS net_qty
    FROM Trades
    GROUP BY CAST(trade_datetime AS date), asset
),
positions AS (
    SELECT
        p.[date],
        p.asset,
        SUM(COALESCE(t.net_qty,0)) OVER (
            PARTITION BY p.asset
            ORDER BY p.[date]
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS position_qty,
        p.close_
    FROM Prices p
    LEFT JOIN trades_daily t
        ON t.[date] = p.[date]
        AND t.asset = p.asset
)
SELECT
    [date],
    asset,
    position_qty,
    close_,
    position_qty * close_ AS position_value
FROM positions;


-- Portfolio daily NAV
CREATE OR ALTER VIEW v_portfolio_daily AS
WITH nav_calc AS (
    SELECT
        [date],
        SUM(position_value) AS positions_value
    FROM v_positions_daily
    GROUP BY [date]
)
SELECT
    [date],
    positions_value AS nav,
    positions_value - LAG(positions_value) OVER (ORDER BY [date]) AS pnl_daily,
    positions_value - FIRST_VALUE(positions_value) OVER (ORDER BY [date]) AS pnl_cum,
    0 AS cash
FROM nav_calc;


-- Exposure view
CREATE OR ALTER VIEW v_exposure_daily AS
SELECT
    p.[date],
    p.asset,
    p.position_value,
    d.nav,
    CASE 
        WHEN d.nav = 0 THEN NULL
        ELSE CAST(p.position_value / d.nav AS decimal(19,6))
    END AS exposure_pct
FROM v_positions_daily p
JOIN v_portfolio_daily d
  ON d.[date] = p.[date];
