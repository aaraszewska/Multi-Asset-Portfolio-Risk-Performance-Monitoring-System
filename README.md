# Multi-Asset-Portfolio-Risk-Performance-Monitoring-System

# Multi-Asset Portfolio Risk & Performance Monitoring System

## Overview
This project simulates a portfolio monitoring and risk analytics system similar to those used in asset management, fintech and trading environments.

The portfolio consists of:
- Bitcoin (BTC)
- Ethereum (ETH)
- Gold (GOLD)
- NASDAQ Index

## Features
- Daily NAV and PnL calculation
- Asset exposure monitoring
- 30-day rolling volatility
- Maximum drawdown
- Annualised return
- Sharpe ratio

## Tech Stack
- SQL Server (Window Functions)
- Python (NumPy, Pandas)
- Financial risk modelling concepts

## Risk Metrics Implemented
- Rolling Volatility (30D)
- Maximum Drawdown
- Annualised Return
- Sharpe Ratio

## How to Recreate
1. Run `schema.sql`
2. Run `views.sql`
3. Run `risk_metrics.sql`
4. Generate sample data using `generate_data.py`
