# Alice v1

Local assistant prototype with wake word + command execution.

## What it does

- Wake-word commands (default: `Alice`)
- Voice input mode (via `SpeechRecognition`) or text mode
- Conversational replies for general speech after wake word
- Natural-language intent parsing for command execution (OpenAI-backed)
- Optional desktop interface with animated face model (`--ui`)
- Safe command execution with an allowlist
- Confirmation step for dangerous actions (`run_file`, `stop_process`)
- Process logs written to `logs/`

## Supported commands

- `Alice run <path-to-file>`
- `Alice, can you run the gravity script in examples for me?` (natural-language command)
- `Alice list files in <folder>`
- `Alice open folder <folder>`
- `Alice stop process` or `Alice stop process <pid>`
- `Alice help`
- `Alice hello` (and other basic small talk)
- `Alice <any other phrase>` for chat replies
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

Run with Alice interface window:

```bash
python3 src/main.py --mode voice --ui
```

On macOS, Alice now uses native `say` by default for audible responses.
You can choose a voice with:

```bash
export ALICE_VOICE="Samantha"
```

You can also persist voice/model/api key in `AstroPhysics/Alice/.env.local`.

For more human-sounding voice, enable OpenAI TTS:

```bash
export ALICE_TTS_BACKEND="openai"
export ALICE_OPENAI_TTS_MODEL="gpt-4o-mini-tts"
export ALICE_OPENAI_TTS_VOICE="sage"
```

Run in text-input but spoken-output mode:

```bash
python3 src/main.py --mode text
```

Optional (better chat quality):

```bash
pip install openai
export OPENAI_API_KEY="your_key_here"
```

Natural-language command parsing uses OpenAI when `OPENAI_API_KEY` is set.
OpenAI TTS also uses API credits.

One-shot test:

```bash
python3 src/main.py --mode text --no-tts --once --command "Alice run examples/hello_alice.py"
```

## Safety

Allowed paths are controlled in `config/allowed_paths.json`.
By default, Alice is limited to:

- `/Users/fabiofacin/Documents/Physics-and-Programming`

Any path outside allowlisted roots is blocked.
