import pandas as pd
from sklearnex import sklearn_is_patched
sklearn_is_patched()
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv('late.csv',encoding='latin')


import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=1

features = ['Protocol', 'Length']
X = data[features]
X_encoded = pd.get_dummies(X, columns=['Protocol'])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

input_dim = X_scaled.shape[1]
encoding_dim = 10

# Build and train the improved autoencoder model
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
    tf.keras.layers.Dense(32, activation='relu'),  # Add another hidden layer
    tf.keras.layers.Dense(encoding_dim, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),  # Add a hidden layer for decoding
    tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),  # Add another hidden layer
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32)  # Increase the number of epochs

# Use the trained model for anomaly detection
reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 99.9)  # Adjust the percentile threshold based on your data

# Identify congestion points based on the anomaly scores
anomalypoints = data[mse > threshold]
print("Anomaly points:")
display(anomalypoints)

# Import necessary libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas as pd
from IPython.display import display


# Sample dataset loading (replace 'late.csv' with your file path)
data = pd.read_csv('late.csv', encoding='latin')

# Step 1: Select features
features = data[['Protocol', 'Length']]

# Step 2: Encode the categorical column
features_encoded = pd.get_dummies(features, columns=['Protocol'])

# Step 3: Scale the numerical data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_encoded)

# Step 4: Anomaly detection using IsolationForest
isolation_forest = IsolationForest(contamination=0.0045, random_state=42)  # Adjust contamination parameter as needed
anomaly_scores_if = isolation_forest.fit_predict(features_scaled)

# Step 5: Prepare labels for supervised anomaly detection
labels_rf = np.where(anomaly_scores_if == 1, 1, 0)  # 1 = normal, 0 = anomaly

# Step 6: Train a RandomForestClassifier for anomaly prediction
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(features_scaled, labels_rf)

# Step 7: Predict anomalies using the RandomForestClassifier
anomaly_predictions_rf = random_forest.predict(features_scaled)

# Step 8: Filter out anomalies
anomalies = data[anomaly_predictions_rf == 0]  # 0 indicates anomalies

print("Anomalies:")
display(anomalies)  # This will display a clean, interactive table in Jupyter Notebook

