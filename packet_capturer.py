import pyshark
import pandas as pd
import nest_asyncio
import asyncio

# Allow nested event loops (specific for Jupyter)
nest_asyncio.apply()

# Network interface and BPF filter
interface = 'en0'  # Replace with your interface
bpf_filter = ''  # Optional filter: e.g., 'icmp or tcp or udp'

# Output file
output_csv = 'packets.csv'

# Debug info
print(f"Using interface: {interface}")
print(f"Using BPF filter: {bpf_filter}")

# -----------------------
# Step 1: Capture Packets and Save Directly to CSV
# -----------------------
async def capture_packets_to_csv(packet_limit=10):
    try:
        print("Capturing packets and saving directly to CSV...")

        # Initialize packet capture
        capture = pyshark.LiveCapture(interface=interface, bpf_filter=bpf_filter)
        packet_data = []

        for i, packet in enumerate(capture.sniff_continuously()):
            if i >= packet_limit:
                break

            # Map data to columns expected in late.csv
            pkt_info = {
                'No.': i + 1,  # Assign a packet number
                'Time': getattr(packet, 'sniff_time', None),
                'Source': getattr(packet.ip, 'src', None) if hasattr(packet, 'ip') else None,
                'Destination': getattr(packet.ip, 'dst', None) if hasattr(packet, 'ip') else None,
                'Protocol': getattr(packet, 'highest_layer', None),
                'Length': getattr(packet, 'length', None),
                'Info': packet.summary() if hasattr(packet, 'summary') else None,  # Brief packet info
            }

            packet_data.append(pkt_info)
            print(f"Captured Packet {i + 1}: {pkt_info}")

        # Save to CSV
        df = pd.DataFrame(packet_data)

        # Ensure column order matches late.csv
        column_order = ['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
        df = df[column_order]

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"{packet_limit} packets saved directly to {output_csv}")

    except Exception as e:
        print(f"Error: {e}")


# -----------------------
# Step 2: Run Capture Function
# -----------------------
loop = asyncio.get_event_loop()
if loop.is_running():
    task = loop.create_task(capture_packets_to_csv(packet_limit=10))
else:
    loop.run_until_complete(capture_packets_to_csv(packet_limit=10))
