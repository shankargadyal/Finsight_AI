"""
train_model.py - Offline LSTM training for FinSight AI.

Run this from your machine or a Render one-off/cron job -- NEVER import it
from the Flask app, and never call any function here from a request handler.

Usage:
    python train_model.py AAPL TSLA MSFT NVDA
    python train_model.py AAPL --epochs 40 --period 5y

For each symbol this will save, into saved_models/:
    <SYMBOL>.keras          - the trained model
    <SYMBOL>_scaler.pkl     - the fitted MinMaxScaler
    <SYMBOL>_metrics.json   - validation RMSE / MAE, used by LSTMModel.predict()
                              at inference time so the API doesn't need to
                              recompute them on every request.
"""
import os
import json
import logging
import argparse

import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finsight.train")

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
SEQUENCE_LENGTH = 60
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 32


def fetch_history(symbol, period="5y"):
    import yfinance as yf
    logger.info("[%s] Downloading %s of history...", symbol, period)
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    return df["Close"].dropna().values.reshape(-1, 1)


def build_sequences(scaled_prices, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled_prices)):
        X.append(scaled_prices[i - seq_len:i, 0])
        y.append(scaled_prices[i, 0])
    return np.array(X), np.array(y)


def build_model(seq_len):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len, 1)),
        layers.LSTM(50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_symbol(symbol, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, period="5y"):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    symbol = symbol.upper()
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    prices = fetch_history(symbol, period=period)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    split_idx = int(len(scaled) * 0.85)
    train_data = scaled[:split_idx]
    test_data = scaled[split_idx - SEQUENCE_LENGTH:]  # keep lookback context for test sequences

    X_train, y_train = build_sequences(train_data, SEQUENCE_LENGTH)
    X_test, y_test = build_sequences(test_data, SEQUENCE_LENGTH)

    if len(X_train) < 50:
        raise RuntimeError(
            f"[{symbol}] Not enough history to train reliably ({len(X_train)} sequences)"
        )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model(SEQUENCE_LENGTH)

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    logger.info("[%s] Training on %d sequences...", symbol, len(X_train))
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    mae = float(mean_absolute_error(actual, preds))
    logger.info("[%s] Validation RMSE=%.4f MAE=%.4f", symbol, rmse, mae)

    model_path = os.path.join(SAVED_MODELS_DIR, f"{symbol}.keras")
    scaler_path = os.path.join(SAVED_MODELS_DIR, f"{symbol}_scaler.pkl")
    metrics_path = os.path.join(SAVED_MODELS_DIR, f"{symbol}_metrics.json")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump({
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "trained_on_period": period,
            "sequence_length": SEQUENCE_LENGTH,
        }, f, indent=2)

    logger.info("[%s] Saved model -> %s", symbol, model_path)
    return {"symbol": symbol, "rmse": rmse, "mae": mae}


def main():
    parser = argparse.ArgumentParser(description="Offline LSTM trainer for FinSight AI")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to train, e.g. AAPL TSLA")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        try:
            results.append(train_symbol(symbol, args.epochs, args.batch_size, args.period))
        except Exception as e:
            logger.error("[%s] Training failed: %s", symbol, e)

    logger.info("Training run complete:")
    for r in results:
        logger.info("  %s: RMSE=%.4f MAE=%.4f", r["symbol"], r["rmse"], r["mae"])


if __name__ == "__main__":
    main()
