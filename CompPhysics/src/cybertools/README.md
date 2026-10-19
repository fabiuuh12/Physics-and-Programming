# CyberTools

Security, network, forensics, camera, and file utilities within the same CompPhysics collection. Browse by task below.

From the repository root:

```bash
python3 explore.py list cybertools
python3 explore.py info "cybertools/network/Certificate Expiry Checker.py"
python3 explore.py run "cybertools/network/Certificate Expiry Checker.py" --help
```

Some scripts prompt for inputs; others accept command-line arguments. Read their inputs and source before running file-changing operations.

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

Many utilities use only the standard library. Shared dependencies are in [`config/requirements.txt`](../../config/requirements.txt); see [Getting started](../../docs/GETTING_STARTED.md). The pose recorder uses `resources/models/pose_landmarker_lite.task` and stores its default database under `output/data/`.
