from scapy.all import *
import os

db_type = "postgres_13_gcp"
scenario_id = 2
dump_id = 1
port = 5432

def packet_callback(packet):
  """
  Callback function to process both incoming and outgoing packets on port {port}, 
  and display the hexdump of the application layer data.

  Args:
    packet: The intercepted packet.
  """
  
  global dump_id
  
  if TCP in packet and (packet[TCP].sport == port or packet[TCP].dport == port):
    print("Packet Summary:")
    print(packet.summary())

    print("\nApplication Layer Hexdump:")
    try:
      app_layer_data = bytes(packet[TCP].payload)
      if app_layer_data:
        print(hexdump(app_layer_data))
        path = f"/Users/raresraf/code/project-martial/experimental/network/{db_type}/scenarios/scenario_{scenario_id}/dump_{dump_id}.bin"
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as file:
            file.write(app_layer_data)
            dump_id = dump_id + 1
      else:
        print("No application layer data found.")
    except Exception as e:
      print(f"Error processing application layer data: {e}")
    print("-" * 30)


# Start sniffing packets on a port (e.g. 3306), both incoming and outgoing.
sniff(prn=packet_callback, filter=f"tcp port {port}")

