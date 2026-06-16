import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==============================
# CONFIGURATION
# ==============================
STOCK = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32

# ==============================
# DOWNLOAD DATA
# ==============================
print("Downloading stock data...")

data = yf.download(
    STOCK,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)

close_prices = data['Close'].values.reshape(-1, 1)

# ==============================
# SCALE DATA
# ==============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ==============================
# CREATE SEQUENCES
# ==============================
X = []
y = []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i - LOOKBACK:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ==============================
# TRAIN TEST SPLIT
# ==============================
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==============================
# BUILD MODEL
# ==============================
print("Building model...")

model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ==============================
# TRAIN
# ==============================
print("Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==============================
# PREDICTIONS
# ==============================
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# ==============================
# EVALUATION
# ==============================
rmse = np.sqrt(
    np.mean((predictions - actual) ** 2)
)

print(f"\nRMSE: {rmse:.2f}")

# ==============================
# NEXT DAY PREDICTION
# ==============================
last_60_days = scaled_data[-LOOKBACK:]

future_input = np.reshape(
    last_60_days,
    (1, LOOKBACK, 1)
)

future_price = model.predict(future_input)

future_price = scaler.inverse_transform(
    future_price
)

print(
    f"\nPredicted Next Day Price: ${future_price[0][0]:.2f}"
)

# ==============================
# SAVE MODEL
# ==============================
model.save("stock_lstm_model.h5")

print("Model saved as stock_lstm_model.h5")

# ==============================
# PLOT RESULTS
# ==============================

plt.figure(figsize=(14, 6))

plt.plot(
    actual,
    label='Actual Price'
)

plt.plot(
    predictions,
    label='Predicted Price'
)

plt.title(f'{STOCK} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

plt.show()