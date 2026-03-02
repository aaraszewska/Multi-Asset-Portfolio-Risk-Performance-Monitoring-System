-- ==========================================
-- DATABASE SCHEMA
-- Multi-Asset Portfolio Risk Monitoring
-- ==========================================

-- Prices table
IF OBJECT_ID('Prices','U') IS NOT NULL
    DROP TABLE Prices;

CREATE TABLE Prices (
    [date]  date        NOT NULL,
    asset   varchar(20) NOT NULL,
    close_  float       NOT NULL,
    CONSTRAINT PK_Prices PRIMARY KEY ([date], asset)
);

-- Trades table
IF OBJECT_ID('Trades','U') IS NOT NULL
    DROP TABLE Trades;

CREATE TABLE Trades (
    trade_id        int IDENTITY(1,1) PRIMARY KEY,
    trade_datetime  datetime2(0)      NOT NULL,
    asset           varchar(20)       NOT NULL,
    side            varchar(4)        NOT NULL CHECK (side IN ('BUY','SELL')),
    quantity        decimal(18,8)     NOT NULL,
    price           decimal(19,4)     NOT NULL,
    fees            decimal(19,4)     NOT NULL DEFAULT 0,
    created_at      datetime2(0)      NOT NULL DEFAULT SYSDATETIME()
);
