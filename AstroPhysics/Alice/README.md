# Alice (Python Rewrite)

Alice is now implemented in Python in this folder.

## What is included

- Rule-based intent parsing (`run`, `list files`, `open folder`, `stop process`, `time`, `date`, `search web`, `research`, `remember`, `recall`, `chat`)
- Local memory store (TSV file, default: `data/alice_memory.tsv`)
- Safe command execution with allowlisted roots (`config/allowed_paths.json`)
- Python and shell file execution support (`.py`, `.sh`, `.bash`, or executable files)
- Optional chat backend via Ollama or OpenAI using environment variables
- macOS spoken replies via `say` (disable with `--no-tts`)
- Voice/UI/face-tracking interfaces are present as stubs in Python and currently report unavailable by default

## Run

```bash
cd AstroPhysics/Alice
python3 main.py --mode text
```

One-shot command:

```bash
python3 main.py --mode text --once --command "Alice run examples/hello_alice.py"
```

Wake word behavior:

```bash
# wake word optional
python3 main.py --mode text

# require wake word
python3 main.py --mode text --require-wake
```

## Configuration

Allowed roots and logs:

- `config/allowed_paths.json`

LLM backend:

```bash
# local
export ALICE_LLM_BACKEND=ollama
export ALICE_OLLAMA_HOST=http://127.0.0.1:11434
export ALICE_OLLAMA_MODEL=qwen2.5:3b

# cloud
export ALICE_LLM_BACKEND=openai
export OPENAI_API_KEY=your_key_here
export ALICE_OPENAI_MODEL=gpt-4o-mini
```

Memory path:

```bash
export ALICE_MEMORY_DB=/absolute/path/to/alice_memory.tsv
```

Speech tuning (optional):

```bash
export ALICE_TTS_RATE=185
export ALICE_VOICE=Samantha
# set to 0 to disable automatic voice selection when ALICE_VOICE is empty
export ALICE_TTS_AUTO_VOICE=1
```

## Notes

- `--mode voice`, `--ui`, and camera options are accepted for compatibility, but the current Python rewrite runs in text mode unless you replace the provided stubs.
- For TTS on macOS, allow your terminal app to use speech services if prompted.
