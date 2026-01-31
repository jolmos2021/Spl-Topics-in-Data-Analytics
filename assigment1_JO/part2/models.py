from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_models(df):
    features = ["close", "volume", "ma_10", "rsi"]
    df = df.dropna()

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    return {
        "Linear Regression RMSE": lr_rmse,
        "Random Forest RMSE": rf_rmse
    }