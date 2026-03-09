# Alice (Python Rewrite)

Alice is now implemented in Python in this folder.

## What is included

- Rule-based intent parsing (`run`, `list files`, `open folder`, `stop process`, `time`, `date`, `search web`, `research`, `remember`, `recall`, `chat`)
- Local memory store (SQLite, default: `data/alice_memory.db`) with auto-fact extraction, upsert updates, and duplicate suppression
- Safe command execution with allowlisted roots (`config/allowed_paths.json`)
- Python and shell file execution support (`.py`, `.sh`, `.bash`, or executable files)
- Optional chat backend via Ollama or OpenAI using environment variables
- macOS spoken replies via `say` (disable with `--no-tts`)
- Built-in classic early-Python Alice face model (restored) with a blue themed background
- Emotion engine with a large emotion set (valence/arousal + blended top emotions) driving expressions
- Webcam perception (OpenCV): face tracking + heuristic scene understanding (`office`, `outdoor`, `bedroom`, etc.)
- Multi-face breakdown: when multiple faces are visible, Alice reports each face separately (position, distance, lighting, expression cue, estimated skin tone/hair color/eye color)
- Scene-aware replies: ask `can you see me` or `look around` / `describe the room`
- Safe self-update mode: `update yourself for <goal>` (confirmation required, allowlisted files, compile check + rollback)
- Feedback learning signals: positive/negative user feedback is stored in memory and reused by autonomous planning
- Autonomous drives loop: Alice can choose between self-update, web research, and reflection based on internal drive weights
- Decision learning logs: autonomous decisions/outcomes are stored with numeric rewards and used to adapt strategy
- Structured internal learning: autonomous question/answer/source/confidence/last-checked records are stored in SQLite
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

Autonomous self-improvement mode:

```bash
# default in interactive mode: on
python3 main.py --mode text

# tune cadence
python3 main.py --mode text --autonomy-interval 180 --autonomy-cooldown 180 --autonomy-max-updates 4 --autonomy-warmup 30 --autonomy-exploration 0.22

# autonomous web learning controls
python3 main.py --mode text --autonomous-web --autonomy-web-cooldown 120 --autonomy-max-web-researches 8

# autonomous self-talk while idle (speaks and logs internal reflections)
python3 main.py --mode text --autonomous-self-talk --autonomy-self-talk-interval 90

# keep autonomous learning quiet for a short window after you talk
python3 main.py --mode text --autonomy-presence-grace 20

# disable autonomous web learning, keep other autonomous behavior
python3 main.py --mode text --no-autonomous-web

# disable autonomous self-talk
python3 main.py --mode text --no-autonomous-self-talk

# disable autonomous updates
python3 main.py --mode text --no-autonomous
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
export ALICE_MEMORY_DB=/absolute/path/to/alice_memory.db
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
- On first run with SQLite memory, Alice auto-imports legacy `alice_memory.tsv` entries into `alice_memory.db` if the DB is empty.
- If left running with autonomy enabled, Alice continues background learning ticks even without live conversation.
