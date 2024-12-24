import pyshark
import pandas as pd
import nest_asyncio
import asyncio

# Allow nested event loops (specific for Jupyter)
nest_asyncio.apply()

# Network interface and BPF filter
interface = '\\Device\\NPF_{79DB1438-2663-4543-8C8B-21F133BCFE3D}'  # Ganti dengan interface Anda
bpf_filter = 'icmp or tcp port 80 or tcp or udp'  # Filter: ICMP, HTTP, TCP, UDP traffic

# Output file
output_csv = 'packets_data.csv'

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
            
            pkt_info = {
                'Timestamp': getattr(packet, 'sniff_time', None),
                'Source_IP': getattr(packet.ip, 'src', None) if hasattr(packet, 'ip') else None,
                'Destination_IP': getattr(packet.ip, 'dst', None) if hasattr(packet, 'ip') else None,
                'Source_Port': getattr(packet.tcp, 'srcport', None) if hasattr(packet, 'tcp') else 
                               (getattr(packet.udp, 'srcport', None) if hasattr(packet, 'udp') else None),
                'Destination_Port': getattr(packet.tcp, 'dstport', None) if hasattr(packet, 'tcp') else 
                                    (getattr(packet.udp, 'dstport', None) if hasattr(packet, 'udp') else None),
                'Protocol': getattr(packet.highest_layer, 'layer_name', None),
                'Packet_Length': getattr(packet, 'length', None),
            }

            packet_data.append(pkt_info)
            print(f"Captured Packet {i+1}: {pkt_info}")

        # Save to CSV
        df = pd.DataFrame(packet_data)
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
