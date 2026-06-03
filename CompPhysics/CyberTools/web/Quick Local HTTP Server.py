"""Serve a local folder over HTTP."""
import http.server
import os
import socketserver
from pathlib import Path


def main() -> None:
    folder = Path(input("Folder (default .): ").strip() or ".").resolve()
    port = int(input("Port (default 8000): ").strip() or "8000")
    if not folder.is_dir():
        print("Invalid folder")
        return

    os.chdir(folder)
    print(f"Serving {folder} at http://localhost:{port}")
    print("Ctrl+C to stop")
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("Stopped")


if __name__ == "__main__":
    main()

