import pandas as pd

def add_indicators_pandas(df):
    df = df.sort_values(["name", "date"])

    # Moving Average (10 days)
    df["ma_10"] = (
        df.groupby("name")["close"]
        .rolling(window=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # RSI
    delta = df.groupby("name")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(df["name"]).rolling(14).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df["name"]).rolling(14).mean().reset_index(level=0, drop=True)

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df