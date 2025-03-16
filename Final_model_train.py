import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("updated_dataset_with_2023_final_its_eleven.csv")

# Handling categorical features
categorical_cols = ["Region", "Event Name", "Weather"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cats = encoder.fit_transform(df[categorical_cols])


df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.weekday
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

# Selecting numerical features
numerical_cols = ["Denied Rides", "Idle Drivers", "Surge Multiplier", 
                  "Traffic Level", "Avg Trip Fare", "Date", "Time", "Searches Norm"]

# Combine all features
X = np.hstack([df[numerical_cols].values, encoded_cats])
y = df[["Searches", "Available Drivers"]].values  # Predicting Demand & Supply

# Scaling
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Hyperbolic Neural Network with LSTM
input_layer = Input(shape=(1, X_train.shape[2]))
lstm_layer = LSTM(64, activation='tanh', return_sequences=True)(input_layer)
lstm_layer = LSTM(32, activation='tanh')(lstm_layer)
dense_layer = Dense(16, activation='tanh')(lstm_layer)
output_layer = Dense(2, activation='linear')(dense_layer)


model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate Accuracy
mae = np.mean(np.abs(y_test_actual - y_pred), axis=0)
mean_actual = np.mean(y_test_actual, axis=0)
accuracy = 100 - (mae / mean_actual) * 100

print(f"Mean Absolute Error (Searches, Available Drivers): {mae}")
print(f"Accuracy (Searches, Available Drivers): {accuracy}%")

model.save("hyperbolic_ride_demand_model_final.h5")


import matplotlib.pyplot as plt

# Plot Searches (Demand)
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual[:, 0], label="Actual Searches", color='blue', alpha=0.6)
plt.plot(y_pred[:, 0], label="Predicted Searches", color='red', linestyle="dashed", alpha=0.8)
plt.xlabel("Test Samples")
plt.ylabel("Number of Searches")
plt.title("Actual vs. Predicted Searches")
plt.legend()
plt.show()

# Plot Available Drivers (Supply)
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual[:, 1], label="Actual Drivers", color='green', alpha=0.6)
plt.plot(y_pred[:, 1], label="Predicted Drivers", color='orange', linestyle="dashed", alpha=0.8)
plt.xlabel("Test Samples")
plt.ylabel("Number of Available Drivers")
plt.title("Actual vs. Predicted Available Drivers")
plt.legend()
plt.show()
