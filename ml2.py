import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from IPython.display import display

# Load dataset
data = pd.read_csv('late.csv', encoding='latin')

# Step 1: Select features
features = ['Protocol', 'Length']
X = data[features]

# Step 2: Encode the categorical column
X_encoded = pd.get_dummies(X, columns=['Protocol'])

# Step 3: Scale the numerical data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 10

autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(encoding_dim, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32)

# Anomaly detection with autoencoder
reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 99.9)
anomalypoints = data[mse > threshold]

print("Anomaly points:")
display(anomalypoints)

# Anomaly detection with IsolationForest
isolation_forest = IsolationForest(contamination=0.0045, random_state=42)
anomaly_scores_if = isolation_forest.fit_predict(X_scaled)

# Prepare labels for supervised anomaly detection
labels_rf = np.where(anomaly_scores_if == 1, 1, 0)

# Train RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_scaled, labels_rf)

# Predict anomalies
anomaly_predictions_rf = random_forest.predict(X_scaled)

# Filter out anomalies
anomalies = data[anomaly_predictions_rf == 0]

print("Anomalies:")
display(anomalies)
