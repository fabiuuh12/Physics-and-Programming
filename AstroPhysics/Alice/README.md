# Alice (C++ Rewrite)

Alice is now implemented in C++ in this folder.

## What is included

- Rule-based intent parsing (`run`, `list files`, `open folder`, `stop process`, `time`, `date`, `remember`, `recall`, `chat`)
- Local memory store (TSV file, default: `data/alice_memory.tsv`)
- Safe command execution with allowlisted roots (`config/allowed_paths.json`)
- C/C++ compile-and-run support for `.cpp/.cc/.cxx/.c`
- Optional chat backend via Ollama or OpenAI using environment variables

## Build

```bash
cd AstroPhysics/Alice
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/alice --mode text
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

## Notes

- Voice, UI, and camera tracking are not part of this C++ pass yet.
- Python source files were removed from `src/` in this rewrite.
