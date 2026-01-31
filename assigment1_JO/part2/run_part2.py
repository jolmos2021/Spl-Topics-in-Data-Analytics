import pandas as pd
from pathlib import Path
from indicators_pandas import add_indicators_pandas
from models import train_models

print("Running Part 2 pipeline...")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

# 1 Load data (1x only)
df = pd.read_parquet(DATA_DIR / "stocks.parquet")

# 2️ Add technical indicators
df = add_indicators_pandas(df)

# 3️ Prepare target (NEXT DAY CLOSE)
df = df.sort_values(["name", "date"])
df["target"] = (
    df.groupby("name")["close"]
    .shift(-1)
)

# 4️ Remove rows with missing values
df = df.dropna()

# 5️ Train and evaluate models
results = train_models(df)

# 6️ Display results
for model, rmse in results.items():
    print(f"{model}: {rmse:.4f}")

print("Part 2 completed.")