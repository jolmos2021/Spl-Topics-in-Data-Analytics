import polars as pl

def add_indicators_polars(df):
    # Ordenar por ticker y fecha
    df = df.sort(["name", "date"])

    # Media móvil 10 días
    df = df.with_columns([
        pl.col("close")
        .rolling_mean(window_size=10)
        .over("name")
        .alias("ma_10")
    ])

    # Diferencias diarias
    delta = pl.col("close").diff().over("name")

    # Gains y losses usando pl.when
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    # Promedio móvil de gains y losses (14 días)
    avg_gain = gain.rolling_mean(14).over("name")
    avg_loss = loss.rolling_mean(14).over("name")

    # RSI
    rs = avg_gain / avg_loss
    df = df.with_columns([
        (100 - (100 / (1 + rs))).alias("rsi")
    ])

    return df