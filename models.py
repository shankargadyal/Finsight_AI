# =============================================================================
#  models.py  —  FinSight AI · All Three ML Models + Ensemble
# =============================================================================
#
#  Classes:
#    LinearRegressionModel  — sklearn poly regression, fastest
#    ARIMAModel             — statsmodels ARIMA(5,1,0), statistical
#    LSTMModel              — TensorFlow/Keras deep learning, most accurate
#    EnsembleModel          — weighted average (LSTM 50 · ARIMA 30 · LR 20)
#
#  Each model exposes:
#    .predict(close_prices, future_days=7) → dict
#
#  Run standalone:
#    python models.py
# =============================================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

from sklearn.linear_model   import LinearRegression
from sklearn.preprocessing  import MinMaxScaler, PolynomialFeatures
from sklearn.metrics        import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# =============================================================================
#  MODEL 1 — LINEAR REGRESSION
# =============================================================================
class LinearRegressionModel:
    """
    Polynomial degree-2 regression on:
      • Day index (t)
      • 7-day moving average
      • 14-day moving average
      • 21-day moving average

    Returns: predictions, RMSE, MAE
    """

    def predict(self, close_prices: list, future_days: int = 7) -> dict:
        try:
            prices = np.array(close_prices, dtype=float)
            n      = len(prices)
            ser    = pd.Series(prices)

            def ma(w): return ser.rolling(w, min_periods=1).mean().values

            # Feature matrix
            X = np.column_stack([np.arange(n), ma(7), ma(14), ma(21)])
            y = prices

            # Normalise
            scX = MinMaxScaler(); scY = MinMaxScaler()
            Xs  = scX.fit_transform(X)
            ys  = scY.fit_transform(y.reshape(-1,1)).ravel()

            # Polynomial expansion
            poly = PolynomialFeatures(degree=2, include_bias=False)
            Xp   = poly.fit_transform(Xs)

            # Fit
            mdl = LinearRegression().fit(Xp, ys)

            # In-sample RMSE / MAE (last 30 days)
            preds_in = scY.inverse_transform(mdl.predict(Xp).reshape(-1,1)).ravel()
            rmse = float(np.sqrt(mean_squared_error(prices[-30:], preds_in[-30:])))
            mae  = float(mean_absolute_error(prices[-30:], preds_in[-30:]))

            # Forecast
            last_mas = [float(ma(w)[-1]) for w in (7, 14, 21)]
            futX  = np.array([[n + i, *last_mas] for i in range(1, future_days+1)])
            futPs = scY.inverse_transform(
                mdl.predict(poly.transform(scX.transform(futX))).reshape(-1,1)
            ).ravel()
            futPs = np.clip(futPs, prices[-1]*0.5, prices[-1]*2.0)

            print(f"   [LinearReg]  Day-7 → ${futPs[-1]:.2f}  RMSE={rmse:.4f}")
            return {
                "predictions": [round(float(p),2) for p in futPs],
                "rmse":        round(rmse, 4),
                "mae":         round(mae, 4),
                "model":       "Linear Regression",
                "error":       None,
            }
        except Exception as e:
            print(f"   [LinearReg]  ERROR: {e}")
            return {"predictions":[], "rmse":None, "mae":None,
                    "model":"Linear Regression", "error":str(e)}


# =============================================================================
#  MODEL 2 — ARIMA
# =============================================================================
class ARIMAModel:
    """
    ARIMA(5,1,0):
      p=5  auto-regression on last 5 differences
      d=1  first-order differencing for stationarity
      q=0  no MA term (stable, fast)

    Uses last 200 trading days for speed.
    Returns: predictions, AIC, confidence_intervals
    """

    MAX_HISTORY = 200

    def predict(self, close_prices: list, future_days: int = 7) -> dict:
        try:
            prices = np.array(close_prices[-self.MAX_HISTORY:], dtype=float)

            mdl    = ARIMA(prices, order=(5, 1, 0))
            fitted = mdl.fit()
            fc     = fitted.get_forecast(steps=future_days)
            preds  = fc.predicted_mean
            ci     = fc.conf_int(alpha=0.20)   # 80% confidence interval

            preds  = np.clip(preds, prices[-1]*0.5, prices[-1]*2.0)

            # In-sample AIC
            aic = round(float(fitted.aic), 2)

            print(f"   [ARIMA]      Day-7 → ${preds[-1]:.2f}  AIC={aic}")
            return {
                "predictions": [round(float(p),2) for p in preds],
                "aic":         aic,
                "conf_int_lower": [round(float(v),2) for v in ci.iloc[:,0]],
                "conf_int_upper": [round(float(v),2) for v in ci.iloc[:,1]],
                "model":       "ARIMA (5,1,0)",
                "error":       None,
            }
        except Exception as e:
            print(f"   [ARIMA]      ERROR: {e}")
            return {"predictions":[], "aic":None,
                    "conf_int_lower":[], "conf_int_upper":[],
                    "model":"ARIMA (5,1,0)", "error":str(e)}


# =============================================================================
#  MODEL 3 — LSTM
# =============================================================================
class LSTMModel:
    """
    TensorFlow/Keras LSTM:
      Input  → LSTM(64, return_sequences=True) → Dropout(0.2)
             → LSTM(32)                         → Dropout(0.2)
             → Dense(16, relu) → Dense(1)

    Training details:
      • 60-day sliding window
      • MinMaxScaler normalisation
      • 80/20 train-val split
      • EarlyStopping(patience=5)
      • Cached in saved_models/<SYMBOL>.keras
    """

    WINDOW     = 60
    EPOCHS     = 20
    BATCH_SIZE = 32

    def __init__(self, symbol: str = "STOCK"):
        self.symbol   = symbol.upper()
        self.scaler   = MinMaxScaler(feature_range=(0, 1))
        self.model    = None
        os.makedirs("saved_models", exist_ok=True)

    @property
    def _path(self):
        return os.path.join("saved_models", f"{self.symbol}.keras")

    def _sequences(self, scaled: np.ndarray):
        flat = scaled.flatten()
        X, y = [], []
        for i in range(self.WINDOW, len(flat)):
            X.append(flat[i - self.WINDOW : i])
            y.append(flat[i])
        return np.array(X).reshape(-1, self.WINDOW, 1), np.array(y)

    def _build(self):
        m = Sequential([
            Input(shape=(self.WINDOW, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer="adam", loss="mean_squared_error")
        return m

    def predict(self, close_prices: list, future_days: int = 7) -> dict:
        try:
            prices = np.array(close_prices, dtype=float).reshape(-1, 1)
            if len(prices) < self.WINDOW + 10:
                raise ValueError(f"Need ≥ {self.WINDOW+10} data points.")

            scaled = self.scaler.fit_transform(prices)

            # Load or train
            if os.path.exists(self._path):
                print(f"   [LSTM]       Loading cached model for {self.symbol} …")
                self.model = tf.keras.models.load_model(self._path)
            else:
                print(f"   [LSTM]       Training {self.symbol} (~30 sec) …")
                X, y = self._sequences(scaled)
                split = int(len(X) * 0.8)
                self.model = self._build()
                self.model.fit(
                    X[:split], y[:split],
                    epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                    validation_data=(X[split:], y[split:]),
                    callbacks=[EarlyStopping(monitor="val_loss",
                                             patience=5, restore_best_weights=True)],
                    verbose=1,
                )
                self.model.save(self._path)
                print(f"   [LSTM]       Saved → {self._path}")

            # Test-set RMSE
            X_all, y_all = self._sequences(scaled)
            split  = int(len(X_all) * 0.8)
            y_pred = self.model.predict(X_all[split:], verbose=0).flatten()
            y_true = self.scaler.inverse_transform(y_all[split:].reshape(-1,1)).ravel()
            y_pred_inv = self.scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_inv)))
            mae  = float(mean_absolute_error(y_true, y_pred_inv))

            # Autoregressive forecast
            seed   = scaled[-self.WINDOW:].flatten().tolist()
            future = []
            for _ in range(future_days):
                inp = np.array(seed[-self.WINDOW:]).reshape(1, self.WINDOW, 1)
                nxt = float(self.model.predict(inp, verbose=0)[0][0])
                future.append(nxt)
                seed.append(nxt)

            preds = self.scaler.inverse_transform(
                np.array(future).reshape(-1, 1)
            ).flatten()
            last  = float(close_prices[-1])
            preds = np.clip(preds, last * 0.5, last * 2.0)

            print(f"   [LSTM]       Day-7 → ${preds[-1]:.2f}  RMSE={rmse:.4f}")
            return {
                "predictions": [round(float(p),2) for p in preds],
                "rmse":        round(rmse, 4),
                "mae":         round(mae, 4),
                "model":       "LSTM",
                "error":       None,
            }
        except Exception as e:
            print(f"   [LSTM]       ERROR: {e}")
            return {"predictions":[], "rmse":None, "mae":None,
                    "model":"LSTM", "error":str(e)}


# =============================================================================
#  ENSEMBLE — weighted average (LSTM 50% · ARIMA 30% · LinReg 20%)
# =============================================================================
class EnsembleModel:
    """
    Runs all three models and combines via weighted average.

    Weights (configurable):
        LSTM            50%
        ARIMA           30%
        Linear Regression 20%
    """

    WEIGHTS = {"lstm": 0.50, "arima": 0.30, "linear_reg": 0.20}

    def predict(self, symbol: str, close_prices: list, future_days: int = 7) -> dict:
        print(f"\n{'═'*52}")
        print(f"  EnsembleModel → {symbol}  ({len(close_prices)} bars)")
        print(f"{'═'*52}")

        last = float(close_prices[-1])

        print("\n  [1/3] Linear Regression")
        lr = LinearRegressionModel().predict(close_prices, future_days)

        print("\n  [2/3] ARIMA")
        ar = ARIMAModel().predict(close_prices, future_days)

        print("\n  [3/3] LSTM")
        ls = LSTMModel(symbol).predict(close_prices, future_days)

        # Weighted ensemble
        results = {"linear_reg": lr, "arima": ar, "lstm": ls}
        valid   = {k: v for k, v in results.items() if not v["error"]
                   and len(v["predictions"]) == future_days}

        if valid:
            total_w  = sum(self.WEIGHTS[k] for k in valid)
            ensemble = []
            for i in range(future_days):
                val = sum(self.WEIGHTS[k] * valid[k]["predictions"][i]
                          for k in valid) / total_w
                ensemble.append(round(val, 2))
        else:
            ensemble = []

        # Trend
        trend = ("up" if ensemble and ensemble[-1] > last else "down") if ensemble else "neutral"

        # Confidence score (0–95)
        agree = sum(
            1 for r in results.values()
            if not r["error"] and r["predictions"]
            and (r["predictions"][-1] > last) == (trend == "up")
        )
        move  = abs((ensemble[-1] - last) / last * 100) if ensemble else 0
        conf  = round(min(40 + agree * 15 + min(move, 10) * 1.5, 95), 1)

        print(f"\n  Ensemble  → {ensemble}")
        print(f"  Trend={trend.upper()}  Confidence={conf}%")
        print(f"{'═'*52}\n")

        return {
            "linear_reg": lr,
            "arima":      ar,
            "lstm":       ls,
            "ensemble":   ensemble,
            "trend":      trend,
            "confidence": conf,
            "last_close": last,
        }


# =============================================================================
#  STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    prices = [150.0]
    for _ in range(399):
        prices.append(round(prices[-1] * (1 + np.random.normal(0.0003, 0.013)), 2))

    result = EnsembleModel().predict("TEST", prices, 7)

    print("\n─── RESULTS ───────────────────────────────────────")
    for k in ("linear_reg", "arima", "lstm"):
        m = result[k]
        print(f"{m['model']:22s}: {m['predictions']}  RMSE={m['rmse']}")
    print(f"{'Ensemble':22s}: {result['ensemble']}")
    print(f"Trend={result['trend'].upper()}  Confidence={result['confidence']}%")
