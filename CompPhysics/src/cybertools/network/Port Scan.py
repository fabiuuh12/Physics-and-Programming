import socket

def scan_ports(target, ports):
    print(f"Scanning {target}...")
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex((target, port))
            if result == 0:
                print(f"[+] Port {port} is OPEN")
            s.close()
        except socket.error:
            print(f"[!] Couldn't connect to port {port}")

if __name__ == "__main__":
    target_host = input("Enter target IP or domain: ").strip()
    
    # You can change or expand this list
    ports_to_scan = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3306, 3389]
    
    scan_ports(target_host, ports_to_scan)
