import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# -------------------------
# Configuración de página
# -------------------------
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")
st.write("Next-day closing price prediction using historical stock data.")

# -------------------------
# Cargar datos
# -------------------------
DATA_DIR = Path(__file__).resolve().parent / ".." / "data"

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "stocks.parquet")

df = load_data()

# -------------------------
# Seleccionar ticker
# -------------------------
tickers = sorted(df["name"].unique())
ticker = st.selectbox("Select a company ticker:", tickers)

# Filtrar datos del ticker seleccionado
df_ticker = df[df["name"] == ticker].sort_values("date")

# -------------------------
# Feature engineering
# -------------------------
df_ticker["ma_10"] = df_ticker["close"].rolling(10).mean()

delta = df_ticker["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df_ticker["rsi"] = 100 - (100 / (1 + rs))

df_ticker["target"] = df_ticker["close"].shift(-1)
df_ticker = df_ticker.dropna()

features = ["close", "volume", "ma_10", "rsi"]
X = df_ticker[features]
y = df_ticker["target"]

# -------------------------
# Train-test split
# -------------------------
split = int(len(df_ticker) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -------------------------
# Entrenar modelo
# -------------------------
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Predicciones
# -------------------------
predictions = model.predict(X_test)

# -------------------------
# Graficar resultados
# -------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_ticker["date"].iloc[split:], y_test.values, label="Actual Price")
ax.plot(df_ticker["date"].iloc[split:], predictions, label="Predicted Price")

ax.set_title(f"{ticker} - Actual vs Predicted Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)