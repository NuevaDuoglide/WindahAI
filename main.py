from Bestmodel import detect_anomalies
from packet_capturer import capture_packets_to_csv
import asyncio
import pandas as pd

# Parameters
packet_limit = 10  # Number of packets to capture
output_csv = 'packets.csv'

def main():
    # Step 1: Capture packets
    print("Starting packet capture...")
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = loop.create_task(capture_packets_to_csv(packet_limit=packet_limit))
    else:
        loop.run_until_complete(capture_packets_to_csv(packet_limit=packet_limit))
    print(f"Packet capture completed and saved to {output_csv}")

    # Step 2: Load the captured data into a DataFrame (directly from the CSV file if needed)
    print("Loading captured packet data...")
    try:
        data = pd.read_csv(output_csv, encoding='latin')  # Read the data from the captured CSV
        print(f"Data loaded successfully from {output_csv}")
    except FileNotFoundError:
        print(f"File {output_csv} not found, using empty data.")
        data = pd.DataFrame()  # Empty DataFrame if file doesn't exist

    # Step 3: Perform anomaly detection with the captured data
    print("Running anomaly detection...")
    detect_anomalies(data)  # Pass the loaded data directly to detect anomalies
    print("Anomaly detection completed.")

if __name__ == "__main__":
    main()
