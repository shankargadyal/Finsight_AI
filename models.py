"""
models.py - Production ML pipeline for FinSight AI

CRITICAL RULE: model.fit() for the LSTM is NEVER called from this file.
LSTMModel.predict() only loads a pretrained artifact and runs inference.
To (re)train the LSTM, run train_model.py offline (outside the Flask request path).

Linear Regression and ARIMA are still fit per-request because they are cheap,
closed-form / classical-statistics fits (milliseconds, not a training loop) and
carry negligible memory overhead compared to a neural network.
"""
import os
import json
import logging

import numpy as np

logger = logging.getLogger("finsight.models")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


class ModelUnavailableError(Exception):
    """Raised whenever a single model cannot produce a prediction.
    Caught by EnsembleModel so one failure never takes down the whole request."""
    pass


# ─────────────────────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────────────────────
class LinearRegModel:
    def __init__(self, symbol):
        self.symbol = symbol

    def predict(self, close_prices, future_days=7):
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error

            prices = np.asarray(close_prices, dtype=float)
            if len(prices) < 10:
                raise ModelUnavailableError("Not enough data points for linear regression")

            X = np.arange(len(prices)).reshape(-1, 1)
            y = prices

            model = LinearRegression()
            model.fit(X, y)

            in_sample = model.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, in_sample)))

            future_X = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
            future_preds = model.predict(future_X)

            return {
                "predictions": future_preds.tolist(),
                "rmse": round(rmse, 4),
            }
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.warning("[%s] LinearRegModel failed: %s", self.symbol, e)
            raise ModelUnavailableError(str(e))


# ─────────────────────────────────────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────────────────────────────────────
class ARIMAModel:
    def __init__(self, symbol):
        self.symbol = symbol

    def predict(self, close_prices, future_days=7):
        try:
            from statsmodels.tsa.arima.model import ARIMA

            prices = np.asarray(close_prices, dtype=float)
            if len(prices) < 30:
                raise ModelUnavailableError("Not enough data points for ARIMA")

            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()

            forecast = fitted.get_forecast(steps=future_days)
            mean_fc = np.asarray(forecast.predicted_mean)
            conf_int = np.asarray(forecast.conf_int(alpha=0.05))

            return {
                "predictions": mean_fc.tolist(),
                "conf_int_lower": conf_int[:, 0].tolist(),
                "conf_int_upper": conf_int[:, 1].tolist(),
                "aic": round(float(fitted.aic), 2),
            }
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.warning("[%s] ARIMAModel failed: %s", self.symbol, e)
            raise ModelUnavailableError(str(e))


# ─────────────────────────────────────────────────────────────────────────
# LSTM — INFERENCE ONLY
# ─────────────────────────────────────────────────────────────────────────
class LSTMModel:
    """
    Loads a pretrained .keras model + scaler produced by train_model.py.
    predict() NEVER trains. If the artifact is missing, it raises
    ModelUnavailableError with a clear message so the ensemble can skip it
    and the API can (if it's the only model requested) return:
        {"error": "Pretrained model not found"}
    """
    SEQUENCE_LENGTH = 60

    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.model_path = os.path.join(SAVED_MODELS_DIR, f"{self.symbol}.keras")
        self.scaler_path = os.path.join(SAVED_MODELS_DIR, f"{self.symbol}_scaler.pkl")
        self.metrics_path = os.path.join(SAVED_MODELS_DIR, f"{self.symbol}_metrics.json")

    def is_available(self):
        return os.path.exists(self.model_path) and os.path.exists(self.scaler_path)

    def _load(self):
        if not self.is_available():
            raise ModelUnavailableError("Pretrained model not found")

        # Imported lazily so Flask workers that never touch the LSTM path
        # don't pay TensorFlow's import-time memory cost.
        import tensorflow as tf
        import joblib

        model = tf.keras.models.load_model(self.model_path, compile=False)
        scaler = joblib.load(self.scaler_path)
        return model, scaler

    def _load_metrics(self):
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def predict(self, close_prices, future_days=7):
        try:
            model, scaler = self._load()

            prices = np.asarray(close_prices, dtype=float).reshape(-1, 1)
            if len(prices) < self.SEQUENCE_LENGTH:
                raise ModelUnavailableError(
                    f"Need at least {self.SEQUENCE_LENGTH} data points, got {len(prices)}"
                )

            scaled = scaler.transform(prices)
            window = scaled[-self.SEQUENCE_LENGTH:].reshape(1, self.SEQUENCE_LENGTH, 1)

            preds_scaled = []
            current_window = window.copy()
            for _ in range(future_days):
                next_scaled = float(model.predict(current_window, verbose=0)[0][0])
                preds_scaled.append(next_scaled)
                current_window = np.append(
                    current_window[:, 1:, :], [[[next_scaled]]], axis=1
                )

            preds = scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()

            metrics = self._load_metrics()

            return {
                "predictions": preds.tolist(),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
            }
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error("[%s] LSTMModel inference failed: %s", self.symbol, e)
            raise ModelUnavailableError(str(e))


# ─────────────────────────────────────────────────────────────────────────
# Ensemble — degrades gracefully
# ─────────────────────────────────────────────────────────────────────────
class EnsembleModel:
    """
    Runs Linear Regression, ARIMA, and LSTM independently. Any single
    failure (including a missing pretrained LSTM) is caught and the
    remaining models' predictions are reweighted and combined -- the
    request never 500s just because one model isn't ready.
    """
    BASE_WEIGHTS = {"linear_reg": 0.2, "arima": 0.3, "lstm": 0.5}

    def predict(self, symbol, prices, future_days=7):
        results = {}
        errors = {}

        for name, cls in (
            ("linear_reg", LinearRegModel),
            ("arima", ARIMAModel),
            ("lstm", LSTMModel),
        ):
            try:
                results[name] = cls(symbol).predict(prices, future_days)
            except ModelUnavailableError as e:
                logger.info("[%s] %s unavailable: %s", symbol, name, e)
                errors[name] = str(e)
            except Exception as e:  # belt-and-braces: never let one model kill the request
                logger.error("[%s] %s unexpected error: %s", symbol, name, e)
                errors[name] = str(e)

        if not results:
            raise ModelUnavailableError(
                "All models failed - unable to produce a prediction for this symbol"
            )

        survivors = list(results.keys())
        weight_sum = sum(self.BASE_WEIGHTS[m] for m in survivors)
        norm_weights = {m: self.BASE_WEIGHTS[m] / weight_sum for m in survivors}

        ensemble = np.zeros(future_days)
        for name in survivors:
            preds = np.asarray(results[name]["predictions"][:future_days], dtype=float)
            if len(preds) < future_days:
                preds = np.pad(preds, (0, future_days - len(preds)), mode="edge")
            ensemble += norm_weights[name] * preds

        # Confidence reflects both model agreement (not modeled here in detail)
        # and how much of the full ensemble weight is actually backed by real models.
        confidence = round(weight_sum / sum(self.BASE_WEIGHTS.values()) * 100, 1)

        # Trend: does the ensemble's 7-day-out forecast sit above or below the
        # most recent actual close? app.py's _make_recommendation() reads this.
        last_actual = float(np.asarray(prices, dtype=float)[-1])
        trend = "up" if ensemble[-1] > last_actual else "down" if ensemble[-1] < last_actual else "flat"

        return {
            "linear_reg": results.get("linear_reg"),
            "arima": results.get("arima"),
            "lstm": results.get("lstm"),
            "ensemble": ensemble.tolist(),
            "confidence": confidence,
            "trend": trend,
            "models_used": survivors,
            "models_failed": errors,
        }
