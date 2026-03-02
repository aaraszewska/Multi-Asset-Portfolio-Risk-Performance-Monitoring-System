import numpy as np
import pandas as pd

np.random.seed(42)

start = "2022-01-01"
end = "2024-12-31"
dates = pd.date_range(start, end, freq="B")

assets = {
    "BTC": {"start_price": 40000, "vol": 0.6},
    "ETH": {"start_price": 3000, "vol": 0.7},
    "GOLD": {"start_price": 1800, "vol": 0.15},
    "NASDAQ": {"start_price": 14000, "vol": 0.25}
}

mu = 0.08
dt = 1/252

frames = []

for asset, params in assets.items():
    S0 = params["start_price"]
    sigma = params["vol"]
    prices = [S0]

    for _ in range(1, len(dates)):
        shock = np.random.normal(0,1)
        St = prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
        prices.append(St)

    df = pd.DataFrame({
        "date": dates.date,
        "asset": asset,
        "close_": prices
    })

    frames.append(df)

prices_df = pd.concat(frames)
prices_df.to_csv("prices_sample.csv", index=False)

print("Sample price data generated.")
