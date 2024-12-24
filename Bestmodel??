import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from IPython.display import display

# Load training dataset
data = pd.read_csv('/Users/irbad/Documents/Semester5/TUBES/late.csv', encoding='latin')

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
threshold = np.percentile(mse, 95)
anomalypoints = data[mse > threshold]

print("Anomaly points in training data:")
display(anomalypoints)


# Load test dataset
test_data = pd.read_csv('/Users/irbad/Documents/Semester5/TUBES/packets.csv', encoding='latin')
X_train = data[features]
packets_X = test_data[features]

encoder = OneHotEncoder(sparse_output=False)
protocol_encoded = encoder.fit_transform(X_train[['Protocol']])
length_scaled = MinMaxScaler().fit_transform(X_train[['Length']])
X_train_encoded = np.hstack((protocol_encoded, length_scaled))

# Step 2: Encode 'Protocol' and scale 'Length'
packets_protocol_encoded = encoder.transform(packets_X[['Protocol']])
packets_length_scaled = MinMaxScaler().fit_transform(packets_X[['Length']])
packets_X_encoded = np.hstack((packets_protocol_encoded, packets_length_scaled))

# Train IsolationForest on the training data
isolation_forest = IsolationForest(contamination=0.0075, random_state=42)
isolation_forest.fit(X_train_encoded)

# Prepare labels for supervised anomaly detection
anomaly_scores_if = isolation_forest.predict(X_train_encoded)
labels_rf = np.where(anomaly_scores_if == 1, 1, 0)

# Train RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_encoded, labels_rf)

# Apply the same preprocessing steps to test data
X_test = test_data[features]
X_test_encoded = pd.get_dummies(X_test, columns=['Protocol'])

# Align test data with the training data columns
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)
X_test_scaled = scaler.transform(X_test_encoded)

# Anomaly detection on test data using autoencoder
reconstructed_test = autoencoder.predict(X_test_scaled)
mse_test = np.mean(np.power(X_test_scaled - reconstructed_test, 2), axis=1)
test_anomalies = test_data[mse_test > threshold]

print("Anomaly points in test data:")
display(test_anomalies)


# Predict anomalies with the trained RandomForestClassifier
packets_anomaly_predictions_rf = random_forest.predict(packets_X_encoded)
packets_anomalies_rf = test_data[packets_anomaly_predictions_rf == 0]

print("Anomalies in packets.csv using RandomForestClassifier:")
display(packets_anomalies_rf)
