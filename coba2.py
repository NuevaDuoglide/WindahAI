import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pyshark
import nest_asyncio
import asyncio
from IPython.display import display

# Allow nested event loops
nest_asyncio.apply()

# -----------------------
# Step 1: Capture Packets and Save to DataFrame
# -----------------------
async def capture_packets_to_dataframe(packet_limit=10):
    interface = 'Wi-Fi'  # Replace with your network interface
    bpf_filter = ''  # Optional: e.g., 'icmp or tcp or udp'
    
    print("Capturing packets into DataFrame...")
    capture = pyshark.LiveCapture(interface=interface, bpf_filter=bpf_filter)
    packet_data = []
    
    for i, packet in enumerate(capture.sniff_continuously()):
        if i >= packet_limit:
            break
        
        pkt_info = {
            'No.': i + 1,
            'Time': getattr(packet, 'sniff_time', None),
            'Source': getattr(packet.ip, 'src', None) if hasattr(packet, 'ip') else None,
            'Destination': getattr(packet.ip, 'dst', None) if hasattr(packet, 'ip') else None,
            'Protocol': getattr(packet, 'highest_layer', None),
            'Length': getattr(packet, 'length', None),
            'Info': packet.summary() if hasattr(packet, 'summary') else None,
        }
        packet_data.append(pkt_info)
    
    df = pd.DataFrame(packet_data)
    column_order = ['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
    df = df[column_order]
    print(f"{packet_limit} packets captured.")
    return df

# -----------------------
# Step 2: Machine Learning Pipeline
# -----------------------
async def anomaly_detection_pipeline():
    # Capture packets
    data = await capture_packets_to_dataframe(packet_limit=10)
    
    # Feature selection
    features = ['Protocol', 'Length']
    X = data[features]
    X_encoded = pd.get_dummies(X, columns=['Protocol'])
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Autoencoder Model
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
    
    # Anomaly Detection
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies_autoencoder = data[mse > threshold]
    print("Anomaly points detected by Autoencoder:")
    display(anomalies_autoencoder)
    
    # Isolation Forest
    isolation_forest = IsolationForest(contamination=0.0075, random_state=42)
    isolation_forest.fit(X_scaled)
    anomaly_scores_if = isolation_forest.predict(X_scaled)
    labels_rf = np.where(anomaly_scores_if == 1, 1, 0)
    
    # Random Forest Classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_scaled, labels_rf)
    predictions_rf = random_forest.predict(X_scaled)
    anomalies_rf = data[predictions_rf == 0]
    print("Anomalies detected by RandomForestClassifier:")
    display(anomalies_rf)

# -----------------------
# Run the Pipeline
# -----------------------
loop = asyncio.get_event_loop()
if loop.is_running():
    task = loop.create_task(anomaly_detection_pipeline())
else:
    loop.run_until_complete(anomaly_detection_pipeline())
