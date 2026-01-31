import pandas as pd
import time
import os
from pathlib import Path

# Directorio del archivo actual (part1/)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

DATA_PATH = DATA_DIR / "all_stocks_5yr.csv"

# Read CSV
start = time.time()
df = pd.read_csv(DATA_PATH)
csv_read_time = time.time() - start

# Save parquet
df.to_parquet(DATA_DIR / "stocks.parquet", engine="pyarrow")
df.to_parquet(
    DATA_DIR / "stocks_snappy.parquet",
    engine="pyarrow",
    compression="snappy"
)

# Read parquet
start = time.time()
pd.read_parquet(DATA_DIR / "stocks.parquet")
parquet_read_time = time.time() - start

# File sizes
csv_size = os.path.getsize(DATA_PATH) / (1024 * 1024)
parquet_size = os.path.getsize(DATA_DIR / "stocks.parquet") / (1024 * 1024)
snappy_size = os.path.getsize(DATA_DIR / "stocks_snappy.parquet") / (1024 * 1024)

print("CSV read time:", csv_read_time)
print("Parquet read time:", parquet_read_time)
print("CSV size (MB):", csv_size)
print("Parquet size (MB):", parquet_size)
print("Parquet Snappy size (MB):", snappy_size)