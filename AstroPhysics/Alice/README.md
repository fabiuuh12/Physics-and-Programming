# Alice v1

Local assistant prototype with wake word + command execution.

## What it does

- Wake-word commands (default: `Alice`)
- Voice input mode with local transcription (`faster-whisper`) when available
- Conversational replies for general speech after wake word
- Natural-language intent parsing for command execution (local Ollama-first)
- Optional desktop interface with expressive animated face model (`--ui`)
- Optional camera face tracking so Alice can look at you (`--camera`)
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
- `Alice can you see me?` (camera-aware reply when `--camera` is enabled)
- `Alice can you see my hand?` (responds naturally, but hand/object detection is not implemented yet)
- `Alice exit`

## Quick start

```bash
cd AstroPhysics/Alice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local AI setup (recommended, no API cost):

```bash
# Install Ollama app first: https://ollama.com/download
ollama pull qwen2.5:3b
```

Set local-first env (persist in `.env.local` or export in shell):

```bash
export ALICE_LLM_BACKEND=ollama
export ALICE_OLLAMA_MODEL=qwen2.5:3b
export ALICE_STT_BACKEND=faster_whisper
export ALICE_WHISPER_MODEL=small
```

If Ollama is installed but not already running, Alice will try to auto-start it.
If `ollama` is not in your PATH, set:

```bash
export ALICE_OLLAMA_CMD=/Applications/Ollama.app/Contents/Resources/ollama
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

Run with face tracking:

```bash
python3 src/main.py --mode voice --ui --camera --camera-owner "Fabio"
```

On macOS, Alice now uses native `say` by default for audible responses.
You can choose a voice with:

```bash
export ALICE_VOICE="Samantha"
```

You can also persist voice/model settings in `AstroPhysics/Alice/.env.local`.

Optional local TTS with Piper:

```bash
# Install Piper CLI separately and download a model (.onnx), then:
export ALICE_TTS_BACKEND=piper
export ALICE_PIPER_MODEL=/absolute/path/to/voice-model.onnx
```

Optional cloud fallback (OpenAI):

```bash
export ALICE_LLM_BACKEND=openai
export ALICE_TTS_BACKEND="openai"
export ALICE_OPENAI_TTS_MODEL="gpt-4o-mini-tts"
export ALICE_OPENAI_TTS_VOICE="sage"
export OPENAI_API_KEY="your_key_here"
```

Run in text-input but spoken-output mode:

```bash
python3 src/main.py --mode text
```

When Ollama is running, chat+intent are local by default.

One-shot test:

```bash
python3 src/main.py --mode text --no-tts --once --command "Alice run examples/hello_alice.py"
```

## Safety

Allowed paths are controlled in `config/allowed_paths.json`.
By default, Alice is limited to:

- `/Users/fabiofacin/Documents/Physics-and-Programming`

Any path outside allowlisted roots is blocked.
