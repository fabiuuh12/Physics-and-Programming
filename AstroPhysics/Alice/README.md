# Alice (C++ Rewrite)

Alice is implemented in C++ in this folder.

## What is included

- Rule-based intent parsing (`run`, `list files`, `open folder`, `stop process`, `time`, `date`, `remember`, `recall`, `chat`)
- Local memory store (TSV file, default: `data/alice_memory.tsv`)
- Safe command execution with allowlisted roots (`config/allowed_paths.json`)
- C/C++ compile-and-run support for `.cpp/.cc/.cxx/.c`
- Optional chat backend via Ollama or OpenAI using environment variables
- macOS UI mode (`--ui`) with animated Alice face + chat log
- macOS spoken replies via `say` (disable with `--no-tts`)
- macOS voice input mode (`--mode voice`) via Apple Speech framework
- macOS camera face tracking so Alice's eyes follow your face in UI mode

## Build

```bash
cd AstroPhysics/Alice
# if CMake is installed:
cmake -S . -B build
cmake --build build -j

# fallback without CMake:
clang++ -std=c++20 -O2 -fobjc-arc -Iinclude \
  src/main.cpp src/string_utils.cpp src/config.cpp src/intent.cpp \
  src/memory_store.cpp src/executor.cpp src/llm_client.cpp src/brain.cpp src/ui_macos.mm src/voice_listener_macos.mm src/face_tracker_macos.mm \
  -framework AppKit -framework AVFoundation -framework Speech -framework Vision -framework CoreMedia -framework CoreVideo \
  -Wl,-sectcreate,__TEXT,__info_plist,resources/AliceInfo.plist \
  -o build/alice
```

## Run

Text mode:

```bash
./build/alice --mode text
```

Voice mode:

```bash
./build/alice --mode voice
```

`--mode voice` enables the UI face by default. Use `--no-ui` to run voice mode without a window.
Voice mode also supports barge-in: start speaking while Alice talks and she will stop TTS and listen.
If you want to disable barge-in (for example to avoid speaker echo interruptions), set `ALICE_BARGE_IN=0`.

UI mode (face-only 3D window + spoken responses):

```bash
./build/alice --mode text --ui
```

UI + voice mode:

```bash
./build/alice --mode voice --ui
```

UI mode without camera tracking:

```bash
./build/alice --mode text --ui --no-camera
```

Disable speech:

```bash
./build/alice --mode text --ui --no-tts
```

One-shot command:

```bash
./build/alice --mode text --once --command "Alice run examples/hello_alice.cpp"
```

Wake word behavior:

```bash
# wake word optional
./build/alice --mode text

# require wake word
./build/alice --mode text --require-wake
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

Compiler flags for launched C/C++ files:

```bash
export ALICE_CXX=clang++
export ALICE_CC=clang
export ALICE_CXXFLAGS="-I/opt/homebrew/include"
export ALICE_LDFLAGS="-L/opt/homebrew/lib"
```

Speech tuning (optional):

```bash
export ALICE_STT_LOCALE="en-US"
export ALICE_STT_ON_DEVICE=1
```

## Notes

- For voice mode on macOS, allow microphone and speech recognition access for your terminal app (Terminal/iTerm) in System Settings > Privacy & Security.
- For face tracking in UI mode, allow camera access for your terminal app in System Settings > Privacy & Security > Camera.
