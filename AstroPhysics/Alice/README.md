# Alice v1

Local assistant prototype with wake word + command execution.

## What it does

- Wake-word commands (default: `Alice`)
- Voice input mode (via `SpeechRecognition`) or text mode
- Safe command execution with an allowlist
- Confirmation step for dangerous actions (`run_file`, `stop_process`)
- Process logs written to `logs/`

## Supported commands

- `Alice run <path-to-file>`
- `Alice list files in <folder>`
- `Alice open folder <folder>`
- `Alice stop process` or `Alice stop process <pid>`
- `Alice help`
- `Alice exit`

## Quick start

```bash
cd AstroPhysics/Alice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run in text mode:

```bash
python3 src/main.py --mode text --no-tts
```

Run in voice mode:

```bash
python3 src/main.py --mode voice
```

One-shot test:

```bash
python3 src/main.py --mode text --no-tts --once --command "Alice run examples/hello_alice.py"
```

## Safety

Allowed paths are controlled in `config/allowed_paths.json`.
By default, Alice is limited to:

- `/Users/fabiofacin/Documents/Physics-and-Programming`

Any path outside allowlisted roots is blocked.
