"""Price prediction models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import add_technical_indicators, infer_trend


@dataclass
class PredictionResult:
    current_price: float
    predicted_price: float
    expected_return_pct: float
    trend: str
    confidence: float
    method: str
    details: dict[str, float | str] = field(default_factory=dict)


def _fallback_predict(feature_frame: pd.DataFrame) -> PredictionResult:
    close = feature_frame["Close"].astype(float).dropna()
    returns = close.pct_change().dropna()
    current_price = float(close.iloc[-1])

    short_momentum = returns.tail(5).mean() if len(returns) >= 5 else 0
    medium_momentum = returns.tail(20).mean() if len(returns) >= 20 else short_momentum
    volatility = returns.tail(20).std() if len(returns) >= 20 else returns.std() if not returns.empty else 0

    daily_signal = (0.6 * short_momentum) + (0.4 * medium_momentum)
    daily_signal = float(np.clip(daily_signal, -0.05, 0.05))
    predicted_price = current_price * (1 + daily_signal)
    expected_return_pct = ((predicted_price / current_price) - 1) * 100
    confidence = float(np.clip(0.75 - (volatility * 8), 0.2, 0.8))

    return PredictionResult(
        current_price=current_price,
        predicted_price=float(predicted_price),
        expected_return_pct=float(expected_return_pct),
        trend=infer_trend(feature_frame),
        confidence=confidence,
        method="EWMA momentum fallback",
        details={
            "short_momentum": float(short_momentum or 0),
            "medium_momentum": float(medium_momentum or 0),
            "volatility": float(volatility or 0),
        },
    )


def _lstm_predict(feature_frame: pd.DataFrame, lookback: int, epochs: int) -> PredictionResult:
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.random.set_seed(42)

    feature_columns = ["Close", "Volume", "MA_20", "RSI_14", "MACD", "Volatility_20"]
    model_frame = feature_frame[feature_columns].astype(float).dropna()

    if len(model_frame) < lookback + 20:
        raise ValueError(f"Need at least {lookback + 20} rows for LSTM training.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(model_frame)

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for index in range(lookback, len(scaled)):
        x_values.append(scaled[index - lookback : index])
        y_values.append(float(scaled[index, 0]))

    x_train = np.asarray(x_values)
    y_train = np.asarray(y_values)

    model = Sequential(
        [
            Input(shape=(lookback, len(feature_columns))),
            LSTM(64, return_sequences=True),
            Dropout(0.15),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        x_train,
        y_train,
        epochs=max(1, int(epochs)),
        batch_size=16,
        verbose=0,
        callbacks=[EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)],
    )

    last_sequence = scaled[-lookback:].reshape(1, lookback, len(feature_columns))
    predicted_scaled_close = float(model.predict(last_sequence, verbose=0)[0][0])

    inverse_vector = np.zeros((1, len(feature_columns)))
    inverse_vector[0, 0] = predicted_scaled_close
    predicted_price = float(scaler.inverse_transform(inverse_vector)[0][0])
    current_price = float(model_frame["Close"].iloc[-1])

    validation_size = min(25, len(x_train))
    validation_predictions = model.predict(x_train[-validation_size:], verbose=0).reshape(-1)
    inverse_validation = np.zeros((validation_size, len(feature_columns)))
    inverse_actual = np.zeros((validation_size, len(feature_columns)))
    inverse_validation[:, 0] = validation_predictions
    inverse_actual[:, 0] = y_train[-validation_size:]
    predicted_validation_close = scaler.inverse_transform(inverse_validation)[:, 0]
    actual_validation_close = scaler.inverse_transform(inverse_actual)[:, 0]
    mape = np.mean(np.abs((actual_validation_close - predicted_validation_close) / actual_validation_close))
    confidence = float(np.clip(1 - mape, 0.25, 0.95))

    return PredictionResult(
        current_price=current_price,
        predicted_price=predicted_price,
        expected_return_pct=((predicted_price / current_price) - 1) * 100,
        trend=infer_trend(feature_frame),
        confidence=confidence,
        method="TensorFlow Keras LSTM",
        details={"lookback_days": lookback, "epochs": int(epochs), "validation_mape": float(mape)},
    )


def predict_next_price(
    history: pd.DataFrame,
    lookback: int = 60,
    epochs: int = 5,
    use_lstm: bool = True,
) -> PredictionResult:
    """Predict the next closing price using LSTM when available, otherwise fallback."""
    feature_frame = add_technical_indicators(history)

    if not use_lstm:
        return _fallback_predict(feature_frame)

    try:
        return _lstm_predict(feature_frame, lookback=lookback, epochs=epochs)
    except Exception as exc:
        result = _fallback_predict(feature_frame)
        result.details["fallback_reason"] = str(exc)
        return result

