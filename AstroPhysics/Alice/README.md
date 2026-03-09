# Alice (Python Rewrite)

Alice is now implemented in Python in this folder.

## What is included

- Rule-based intent parsing (`run`, `list files`, `open folder`, `stop process`, `time`, `date`, `search web`, `research`, `remember`, `recall`, `chat`)
- Local memory store (TSV file, default: `data/alice_memory.tsv`) with auto-fact extraction, upsert updates, and duplicate suppression
- Safe command execution with allowlisted roots (`config/allowed_paths.json`)
- Python and shell file execution support (`.py`, `.sh`, `.bash`, or executable files)
- Optional chat backend via Ollama or OpenAI using environment variables
- macOS spoken replies via `say` (disable with `--no-tts`)
- Built-in classic early-Python Alice face model (restored) with a blue themed background
- Emotion engine with a large emotion set (valence/arousal + blended top emotions) driving expressions
- Webcam perception (OpenCV): face tracking + heuristic scene understanding (`office`, `outdoor`, `bedroom`, etc.)
- Scene-aware replies: ask `can you see me` or `look around` / `describe the room`
- Optional microphone voice input mode via `SpeechRecognition` (if installed)

## Run

```bash
cd AstroPhysics/Alice
python3 main.py --mode text
```

Text mode + face UI:

```bash
python3 main.py --mode text --ui
```

In `--ui` mode, Alice also listens to the microphone when available (hands-free).

Voice mode:

```bash
python3 main.py --mode voice
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
# optional: override macOS default voice
export ALICE_VOICE=Samantha
# speech engine for SpeechRecognition listener: google (default) or whisper
export ALICE_STT_ENGINE=google
```

## Notes

- For TTS on macOS, allow your terminal app to use speech services if prompted.
- For microphone input, install dependencies:

```bash
python3 -m pip install SpeechRecognition sounddevice numpy
# optional alternative backend:
# python3 -m pip install pyaudio
# webcam face tracking:
python3 -m pip install opencv-python
```

- For webcam tracking, allow camera access for Terminal/iTerm in System Settings > Privacy & Security > Camera.
- Scene identification is currently heuristic (OpenCV features + motion/light cues), so treat room labels as estimates.
