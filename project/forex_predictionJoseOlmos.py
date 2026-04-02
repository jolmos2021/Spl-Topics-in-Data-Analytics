# ==========================================
# FOREX PREDICTION SCRIPT - NEXT DAY FORECAST
# ==========================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler                          
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from fredapi import Fred
import time
import warnings
import datetime
from datetime import timedelta

warnings.filterwarnings("ignore")

print("="*60)
print("FOREX PREDICTION SCRIPT - NEXT DAY FORECAST")
print("="*60)

# Current date
today = datetime.datetime.today()
today_str = today.strftime('%Y-%m-%d')

# ==========================================
# 1. DOWNLOAD FOREX DATA
# ==========================================

tickers = ["USDCLP=X", "USDCOP=X", "USDCAD=X"]

def download_retry(ticker, start, end, retries=3, delay=5):
    for i in range(retries):
        try:
            print(f"Downloading {ticker} (attempt {i+1})...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            print(f"{ticker} downloaded. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            time.sleep(delay)
    return None

dfs = []
for t in tickers:
    df_t = download_retry(t, start="2016-01-01", end=today_str)

    if df_t is not None:
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.droplevel(1)

        df_t.reset_index(inplace=True)

        if all(col in df_t.columns for col in ['Date','Open','High','Low','Close']):
            df_t = df_t[['Date','Open','High','Low','Close']]
            df_t['pair'] = t
            dfs.append(df_t)

df = pd.concat(dfs, ignore_index=True)
df.sort_values(['pair','Date'], inplace=True)
df['Close'] = df['Close'].astype(float)

print(f"\n✓ Data range: {df['Date'].min()} to {df['Date'].max()}")

# ==========================================
# 2. TECHNICAL INDICATORS
# ==========================================

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

groups = []
for pair, group in df.groupby("pair"):
    group = group.copy().reset_index(drop=True)
    close = group['Close']

    group["SMA_10"] = close.rolling(10).mean()
    group["SMA_30"] = close.rolling(30).mean()
    group["RSI_14"] = compute_rsi(close)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    group["MACD"] = ema12 - ema26

    group["Volatility_20"] = close.pct_change().rolling(20).std()

    group["Close_lag_1"] = close.shift(1)
    group["Close_lag_2"] = close.shift(2)
    group["Close_lag_3"] = close.shift(3)

    group["Returns_1d"] = close.pct_change()
    group["Returns_5d"] = close.pct_change(5)

    groups.append(group)

df = pd.concat(groups)

# ==========================================
# 3. MACRO DATA
# ==========================================

fred = Fred(api_key="7d5f62ae14c796c6329a30ae837d9a74")

macro_series = {
    "FEDFUNDS": "US_Interest_Rate",
    "DCOILWTICO": "Oil_Price",
    "CPIAUCSL": "US_CPI"
}

date_range = pd.date_range(start="2016-01-01", end=today_str, freq="D")
macro_df = pd.DataFrame(index=date_range)

print("\nDownloading macroeconomic data...")

for code, name in macro_series.items():
    try:
        s = fred.get_series(code).reindex(date_range).ffill()
        macro_df[name] = s
        print(f"✓ {name} loaded")
    except:
        macro_df[name] = np.nan
        print(f"✗ Error loading {name}")

macro_df.reset_index(inplace=True)
macro_df.rename(columns={"index":"Date"}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
macro_df['Date'] = pd.to_datetime(macro_df['Date'])

df = df.merge(macro_df, on="Date", how="left")

# ==========================================
# 4. TARGET VARIABLE
# ==========================================

df["Close_next"] = df.groupby("pair")["Close"].shift(-1)

# Clean dataset
df.dropna(inplace=True)

print(f"\nDataset shape after cleaning: {df.shape}")

# ==========================================
# EXPORT FINAL DATASET
# ==========================================
df.to_csv("final_dataset.csv", index=False)
print("✓ Final dataset exported as 'final_dataset.csv'")

# ==========================================
# 5. MODEL TRAINING
# ==========================================

features = [c for c in df.columns if c not in ["Date","pair","Close","Close_next"]]

X = df[features]
y = df["Close_next"]

# Time-based split (IMPORTANT)
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\nTraining models...")

lgbm = LGBMRegressor(n_estimators=200)
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1)

lgbm.fit(X_train,y_train)
rf.fit(X_train,y_train)

mae_lgbm = mean_absolute_error(y_test, lgbm.predict(X_test))
mae_rf = mean_absolute_error(y_test, rf.predict(X_test))

best_model = lgbm if mae_lgbm < mae_rf else rf
best_name = "LightGBM" if mae_lgbm < mae_rf else "Random Forest"

print(f"\n✓ Best model: {best_name}")

# ==========================================
# 6. FUTURE PREDICTIONS
# ==========================================

def prepare_features(last_row, macro_latest):
    data = {}
    for col in features:
        if col in last_row:
            data[col] = last_row[col]
        elif col in macro_latest:
            data[col] = macro_latest[col].iloc[0]
        else:
            data[col] = 0
    return pd.DataFrame([data])

def predict_future(df, model, pair, days=2):
    pair_data = df[df['pair']==pair].sort_values('Date')

    last_date = pair_data['Date'].iloc[-1]
    print(f"\n{pair} → Last available date: {last_date}")

    last_row = pair_data.iloc[-1]
    macro_latest = macro_df[macro_df['Date']<=last_date].tail(1)

    preds = []
    current_row = last_row.copy()
    current_date = last_date

    for i in range(days):
        next_date = current_date + timedelta(days=1)

        X_pred = prepare_features(current_row, macro_latest)
        pred = model.predict(X_pred)[0]

        preds.append({
            "Date": next_date,
            "Predicted_Close": pred,
            "pair": pair
        })

        current_row['Close'] = pred
        current_date = next_date

    return pd.DataFrame(preds)

# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

future_predictions = pd.concat([
    predict_future(df, best_model, p, 2)
    for p in df['pair'].unique()
])

future_predictions['Date'] = pd.to_datetime(future_predictions['Date'])

# ==========================================
# NEXT DAY PREDICTION
# ==========================================

today_date = datetime.datetime.today().date()
tomorrow_date = today_date + timedelta(days=1)

print("\n" + "="*60)
print("🎯 NEXT DAY PREDICTION")
print("="*60)

for pair in future_predictions['pair'].unique():
    preds = future_predictions[future_predictions['pair']==pair]

    tomorrow_pred = preds[preds['Date'].dt.date == tomorrow_date]

    if not tomorrow_pred.empty:
        val = tomorrow_pred['Predicted_Close'].iloc[0]
        print(f"{pair} ({tomorrow_date}): {val:.4f}")
    else:
        print(f"{pair}: No prediction available for tomorrow")

# ==========================================
# 8. VISUALIZATION
# ==========================================

print("\nGenerating charts...")

plt.style.use('default')

pairs = df['pair'].unique()
fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 4*len(pairs)))

if len(pairs) == 1:
    axes = [axes]

for idx, pair in enumerate(pairs):
    ax = axes[idx]

    hist = df[df['pair']==pair].sort_values('Date').tail(30)
    preds = future_predictions[future_predictions['pair']==pair]

    ax.plot(hist['Date'], hist['Close'], 'b-o', label='Historical')
    ax.plot(preds['Date'], preds['Predicted_Close'], 'r--s', label='Prediction')

    ax.axvline(x=hist['Date'].iloc[-1], color='gray', linestyle=':')

    ax.set_title(f'{pair} - Historical vs Prediction')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

print("\n✓ SCRIPT COMPLETED SUCCESSFULLY")