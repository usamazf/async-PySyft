## Traffic Analyzis with Tshark
TShark is a network protocol analyzer. It lets you capture packet data from a live network, or read packets from a previously saved capture file, either printing a decoded form of those packets to the standard output or writing the packets to a file. TShark's native capture file format is pcapng format, which is also the format used by wireshark and various other tools.

## Installation
```bash
sudo apt-get update -y
sudo apt-get install -y tshark
```

## Usage
Step 1: Start tshark before starting the worker or server.

Step2: Run this command to track a specific port on eth0. The the outup will then be written into the out.pcap file
```bash
sudo tshark -i eth0 -f "port (YOUR PORT!!!)" -w out.pcap

```

After the the training process is done, press ctr+c to end tshark.
To analyze the traffic file, use the capinfos command
```bash
sudo capinfos -A out.pcap
```