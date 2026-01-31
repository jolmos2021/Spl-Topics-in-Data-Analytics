from pathlib import Path
import pandas as pd
import polars as pl
import time
from indicators_pandas import add_indicators_pandas
from indicators_polars import add_indicators_polars

# Definir directorio de datos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

# -------------------------
# Pandas
# -------------------------
start = time.time()
df_pandas = pd.read_parquet(DATA_DIR / "stocks.parquet")
df_pandas = add_indicators_pandas(df_pandas)
pandas_time = time.time() - start

# -------------------------
# Polars
# -------------------------
start = time.time()
df_polars = pl.read_parquet(DATA_DIR / "stocks.parquet")
df_polars = add_indicators_polars(df_polars)
polars_time = time.time() - start

# -------------------------
# Resultados
# -------------------------
print(f"Pandas time: {pandas_time:.3f}s")
print(f"Polars time: {polars_time:.3f}s")