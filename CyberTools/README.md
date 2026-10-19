# CyberTools

Standalone Python utilities, grouped by task. This collection lives beside `CompPhysics/` so physics simulations and general-purpose tools can be browsed separately.

From the repository root:

```bash
python3 explore.py list CyberTools
python3 explore.py info "CyberTools/network/Certificate Expiry Checker.py"
python3 explore.py run "CyberTools/network/Certificate Expiry Checker.py" --help
```

You can also run a script directly: `python3 "CyberTools/network/Certificate Expiry Checker.py" --help`. Some scripts prompt for inputs; others accept command-line arguments. Read the script before running file-changing operations.

| Folder | Purpose |
| --- | --- |
| [auth](auth/) | Password generation, JWT inspection and authentication-log utilities |
| [file_ops](file_ops/) | File inventories, duplicate searches, conversion and renaming |
| [forensics](forensics/) | Indicators, filenames and log analysis |
| [integrity_hashing](integrity_hashing/) | File hashes and integrity checks |
| [network](network/) | DNS, certificates, ports and subdomain utilities |
| [osint](osint/) | OSINT helper |
| [vision](vision/README.md) | Camera-based presence detection and pose observations |
| [web](web/) | HTTP headers, URL heuristics and a local server |

Many utilities use only Python's standard library. Shared third-party dependencies are currently listed in [`CompPhysics/requirements.txt`](../CompPhysics/requirements.txt); install them in a virtual environment as described in [Getting started](../docs/GETTING_STARTED.md). The pose recorder uses the bundled model in `CompPhysics/vision/models/`.
