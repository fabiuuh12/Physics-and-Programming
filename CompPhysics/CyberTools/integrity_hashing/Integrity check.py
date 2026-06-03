import os
import hashlib
import json

def hash_file(filepath):
    sha = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(4096):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        print(f"[!] Could not hash {filepath}: {e}")
        return None

def scan_directory(dir_path):
    file_hashes = {}
    for root, _, files in os.walk(dir_path):
        for name in files:
            path = os.path.join(root, name)
            file_hash = hash_file(path)
            if file_hash:
                file_hashes[path] = file_hash
    return file_hashes

def save_baseline(hashes, filename="baseline.json"):
    try:
        with open(filename, "w") as f:
            json.dump(hashes, f, indent=4)
        print(f"[✓] Baseline saved to {filename}")
    except Exception as e:
        print(f"[!] Could not save baseline: {e}")

def compare_with_baseline(dir_path, baseline_file="baseline.json"):
    if not os.path.exists(baseline_file):
        print("[!] No baseline found. Run a baseline scan first.")
        return

    try:
        with open(baseline_file, "r") as f:
            old_hashes = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load baseline: {e}")
        return

    new_hashes = scan_directory(dir_path)

    added = [f for f in new_hashes if f not in old_hashes]
    removed = [f for f in old_hashes if f not in new_hashes]
    modified = [f for f in new_hashes if f in old_hashes and new_hashes[f] != old_hashes[f]]

    print("\n--- 🔍 File Integrity Report ---")
    if added:
        print(f"[+] Added files:\n  " + "\n  ".join(added))
    if removed:
        print(f"[-] Removed files:\n  " + "\n  ".join(removed))
    if modified:
        print(f"[~] Modified files:\n  " + "\n  ".join(modified))
    if not (added or removed or modified):
        print("[✓] No changes detected. All clear.")

if __name__ == "__main__":
    print("1. Create baseline\n2. Compare with baseline")
    choice = input("Select option (1/2): ").strip()
    folder = input("Enter folder to scan: ").strip()

    if choice == "1":
        print("[*] Scanning and saving baseline...")
        hashes = scan_directory(folder)
        save_baseline(hashes)
    elif choice == "2":
        print("[*] Comparing current folder to baseline...")
        compare_with_baseline(folder)
    else:
        print("[!] Invalid choice.")
