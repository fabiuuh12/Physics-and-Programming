from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .brain import AliceBrain
from .config import AliceConfig, load_config
from .executor import AliceExecutor
from .face_tracker import FaceTracker
from .intent import describe_for_confirmation, parse_confirmation, parse_intent
from .memory_store import MemoryStore
from .string_utils import format_clock_time, format_long_date, join, normalize_text, replace_all, to_lower, trim
from .types import ExecResult, Intent, MemoryItem
from .ui import AliceUI
from .voice_listener import VoiceListener

HELP_TEXT = (
    "Try commands like: run <file>, list files in <folder>, open folder <folder>, "
    "stop process, what time is it, what is today's date, search the web for <topic>, research <topic>, "
    "remember that <fact>, what do you remember about <topic>, help, exit. Wake word is optional."
)


@dataclass
class Args:
    mode: str = "text"
    require_wake: bool = False
    wake_word: str = "alice"
    once: bool = False
    ui: Optional[bool] = None
    camera: bool = True
    camera_index: int = 0
    no_tts: bool = False
    command: Optional[str] = None
    config_path: Path = Path.cwd() / "config" / "allowed_paths.json"


_g_tts_enabled = False
_g_tts_process: Optional[subprocess.Popen[bytes]] = None
_g_tts_voice_name: Optional[str] = None
_g_tts_rate_wpm = 185


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _smalltalk_reply(topic: Optional[str]) -> str:
    key = to_lower(trim(topic or ""))
    if key in {"hello", "hi", "hey"}:
        return "Hey Fabio. I am here. How are you feeling today?"
    if key in {
        "sorry",
        "i'm sorry",
        "im sorry",
        "i am sorry",
        "my mistake",
        "i made a mistake",
    }:
        return "No problem at all. We can keep going."
    if key in {"never mind", "nevermind"}:
        return "No problem. We can switch topics."
    if key in {"good morning", "good afternoon", "good evening"}:
        return "Hello Fabio. Good to hear from you."
    if key == "how are you":
        return "I am doing well. Curious and ready to help with your project."
    if key in {"who are you", "what is your name"}:
        return "I am Alice, your local AI assistant."
    if key in {"thanks", "thank you"}:
        return "You are welcome."
    return "I am here and listening."


def _sanitize_spoken_text(text: str) -> str:
    out = trim(text)
    out = re.sub(r"https?://\S+", " link ", out, flags=re.IGNORECASE)
    out = replace_all(out, "|", ". ")
    out = replace_all(out, "_", " ")
    out = re.sub(r"\s+", " ", out)
    return trim(out)


def _url_encode_query(value: str) -> str:
    return urllib.parse.quote_plus(value)


def _open_url(url: str) -> bool:
    if _command_exists("open"):
        try:
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False
    try:
        return webbrowser.open(url)
    except Exception:
        return False


def _extract_json_text_entries(payload: dict, limit: int) -> list[str]:
    out: list[str] = []

    def push(text: str) -> None:
        t = trim(text)
        if t and t not in out:
            out.append(t)

    def walk(node: object) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            text = node.get("Text")
            if isinstance(text, str):
                push(text)
            topics = node.get("Topics")
            if isinstance(topics, list):
                for child in topics:
                    walk(child)
                    if len(out) >= limit:
                        return
        elif isinstance(node, list):
            for child in node:
                walk(child)
                if len(out) >= limit:
                    return

    walk(payload.get("RelatedTopics", []))
    return out


def _clamp_spoken_summary(text: str, max_chars: int = 380) -> str:
    cleaned = trim(text)
    if len(cleaned) <= max_chars:
        return cleaned
    cleaned = trim(cleaned[:max_chars])
    last_space = cleaned.rfind(" ")
    if last_space > max_chars // 2:
        cleaned = cleaned[:last_space]
    return f"{trim(cleaned)}..."


def _research_web(query: str) -> ExecResult:
    cleaned_query = trim(query)
    if not cleaned_query:
        return ExecResult(False, "Tell me what topic to research.")

    search_url = f"https://duckduckgo.com/?q={_url_encode_query(cleaned_query)}"

    api_url = (
        "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q="
        f"{_url_encode_query(cleaned_query)}"
    )

    try:
        with urllib.request.urlopen(api_url, timeout=18) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        _open_url(search_url)
        return ExecResult(False, f"I could not fetch research results right now, so I opened web results for {cleaned_query}.")

    heading = data.get("Heading", "") if isinstance(data, dict) else ""
    abstract = data.get("AbstractText", "") if isinstance(data, dict) else ""
    answer = data.get("Answer", "") if isinstance(data, dict) else ""
    source_url = data.get("AbstractURL", "") if isinstance(data, dict) else ""
    related = _extract_json_text_entries(data if isinstance(data, dict) else {}, 4)

    snippets: list[str] = []
    if isinstance(answer, str) and trim(answer):
        snippets.append(trim(answer))
    if isinstance(abstract, str) and trim(abstract):
        snippets.append(trim(abstract))
    for item in related:
        if len(snippets) >= 2:
            break
        snippets.append(item)

    if not snippets:
        _open_url(search_url)
        return ExecResult(False, f"I found limited direct data, so I opened web results for {cleaned_query}.")

    title = heading if isinstance(heading, str) and trim(heading) else cleaned_query
    message = f"Here is what I found about {title}: {_clamp_spoken_summary(join(snippets, ' '))}"
    if isinstance(source_url, str) and trim(source_url):
        message += f" Source: {trim(source_url)}."
    return ExecResult(True, message)


def _tts_rate_from_env() -> int:
    raw = os.getenv("ALICE_TTS_RATE", "")
    if raw:
        try:
            value = int(raw)
            if 120 <= value <= 280:
                return value
        except ValueError:
            pass
    return 185


def _installed_say_voices() -> list[str]:
    try:
        proc = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False)
    except OSError:
        return []

    out: list[str] = []
    for line in proc.stdout.splitlines():
        line = trim(line)
        if not line:
            continue
        out.append(line.split()[0])
    return out


def _natural_voice_from_system() -> Optional[str]:
    explicit_voice = os.getenv("ALICE_VOICE")
    if explicit_voice:
        return explicit_voice

    # When no explicit voice is configured, let `say` use macOS system default voice.
    return None


def _configure_tts() -> None:
    global _g_tts_rate_wpm, _g_tts_voice_name
    _g_tts_rate_wpm = _tts_rate_from_env()
    _g_tts_voice_name = _natural_voice_from_system()


def _stop_tts() -> None:
    global _g_tts_process
    if _g_tts_process is None:
        return
    if _g_tts_process.poll() is None:
        _g_tts_process.terminate()
        try:
            _g_tts_process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _g_tts_process.kill()
            _g_tts_process.wait(timeout=1)
    _g_tts_process = None


def _start_tts(spoken: str) -> bool:
    global _g_tts_process
    _stop_tts()
    if not spoken:
        return False

    args = ["say"]
    if _g_tts_voice_name:
        args += ["-v", _g_tts_voice_name]
    args += ["-r", str(_g_tts_rate_wpm), spoken]

    try:
        _g_tts_process = subprocess.Popen(args)
    except OSError:
        _g_tts_process = None
        return False
    return True


def _speak(text: str, ui: Optional[AliceUI]) -> None:
    print(f"Alice> {text}")

    if ui is not None:
        ui.add_message("Alice", text)
        ui.set_state("speaking")
        ui.set_status("Speaking...")
        ui.pump()

    if _g_tts_enabled and _start_tts(_sanitize_spoken_text(text)):
        while _g_tts_process is not None and _g_tts_process.poll() is None:
            if ui is not None:
                ui.pump()
            time.sleep(0.03)

    if ui is not None:
        ui.set_state("idle")
        ui.set_status("Online")


def _load_env_file(file_path: Path) -> None:
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        line = trim(line)
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = trim(key)
        value = trim(value)
        if not key:
            continue
        if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_project_env(project_root: Path) -> None:
    _load_env_file(project_root / ".env")
    _load_env_file(project_root / ".env.local")


def _parse_args(argv: list[str]) -> Args:
    args = Args()
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--mode" and i + 1 < len(argv):
            args.mode = argv[i + 1]
            i += 2
            continue
        if token == "--require-wake":
            args.require_wake = True
            i += 1
            continue
        if token == "--wake-word" and i + 1 < len(argv):
            args.wake_word = argv[i + 1]
            i += 2
            continue
        if token == "--once":
            args.once = True
            i += 1
            continue
        if token == "--command" and i + 1 < len(argv):
            args.command = argv[i + 1]
            i += 2
            continue
        if token == "--config" and i + 1 < len(argv):
            args.config_path = Path(argv[i + 1])
            i += 2
            continue
        if token == "--ui":
            args.ui = True
            i += 1
            continue
        if token == "--no-ui":
            args.ui = False
            i += 1
            continue
        if token == "--no-camera":
            args.camera = False
            i += 1
            continue
        if token == "--camera-index" and i + 1 < len(argv):
            try:
                args.camera_index = int(argv[i + 1])
            except ValueError:
                args.camera_index = 0
            i += 2
            continue
        if token == "--no-tts":
            args.no_tts = True
            i += 1
            continue
        i += 1
    return args


def _extract_memorable_fact(text: str) -> Optional[str]:
    cleaned = trim(text)
    if not cleaned or "?" in cleaned or len(cleaned) < 8 or len(cleaned) > 180:
        return None

    lowered = to_lower(cleaned)
    if lowered.startswith(("i think ", "i guess ", "maybe ", "probably ")):
        return None

    patterns = [
        r"^my\s+(name|birthday|goal|project|major|school|city|hometown)\s+(is|are)\s+.+$",
        r"^my\s+favorite\s+[a-z][a-z\s]{1,20}\s+(is|are)\s+.+$",
        r"^i('m| am)\s+working\s+on\s+.+$",
        r"^i('m| am)\s+(from|in|studying|learning|building)\s+.+$",
        r"^i\s+(study|work\s+on|build|use)\s+.+$",
        r"^i\s+(like|love|prefer|enjoy|hate)\s+.+$",
        r"^call\s+me\s+.+$",
    ]
    if not any(re.match(pattern, cleaned, flags=re.IGNORECASE) for pattern in patterns):
        return None

    while cleaned and (cleaned[-1] in ".!" or cleaned[-1].isspace()):
        cleaned = cleaned[:-1]
    return cleaned


def _format_recalled_memories(query: str, memories: list[MemoryItem]) -> str:
    if not memories:
        return f"I do not have a saved memory about '{query}' yet."
    return f"Here is what I remember about {query}: {join([item.content for item in memories], ' | ')}"


def _execute_intent(intent: Intent, executor: AliceExecutor) -> ExecResult:
    if intent.action == "help":
        return ExecResult(True, HELP_TEXT)
    if intent.action == "greet":
        return ExecResult(True, "I am listening.")
    if intent.action == "smalltalk":
        return ExecResult(True, _smalltalk_reply(intent.target))
    if intent.action == "list_files":
        return executor.list_files(intent.target)
    if intent.action == "open_folder":
        return executor.open_folder(intent.target)
    if intent.action == "run_file":
        return executor.run_file(intent.target)
    if intent.action == "stop_process":
        return executor.stop_process(intent.pid)
    if intent.action == "get_time":
        return ExecResult(True, f"It is {format_clock_time()}.")
    if intent.action == "get_date":
        return ExecResult(True, f"Today is {format_long_date()}.")
    if intent.action == "web_search":
        query = trim(intent.target or "")
        if not query:
            return ExecResult(False, "Tell me what to search for.")
        url = f"https://duckduckgo.com/?q={_url_encode_query(query)}"
        if not _open_url(url):
            return ExecResult(False, "I could not open web search right now.")
        return ExecResult(True, f"Opened web search results for {query}.")
    if intent.action == "web_research":
        return _research_web(intent.target or "")
    if intent.action == "exit":
        return ExecResult(True, "Shutting down.")
    return ExecResult(False, "I did not understand that command. Say 'Alice help'.")


def _handle_utterance(
    utterance: str,
    wake_word: str,
    require_wake: bool,
    executor: AliceExecutor,
    brain: AliceBrain,
    memory_store: MemoryStore,
    voice_mode: bool,
    voice_listener: Optional[VoiceListener],
    ui: Optional[AliceUI],
) -> bool:
    if ui is not None:
        ui.set_state("thinking")
        ui.set_status("Thinking...")

    intent, _matched = parse_intent(utterance, wake_word, require_wake)
    if intent.action == "skip":
        if ui is not None:
            ui.set_state("idle")
            ui.set_status("Online")
        return True

    if intent.action == "remember_memory":
        fact = trim(intent.target or "")
        if not fact:
            _speak("Tell me what to remember.", ui)
        elif memory_store.add(fact, "profile"):
            _speak("Saved. I will remember that.", ui)
        else:
            _speak("I already remember that.", ui)
        return True

    if intent.action == "recall_memory":
        query = trim(intent.target or "me") or "me"
        recalled = memory_store.search(query, 5)
        _speak(_format_recalled_memories(query, recalled), ui)
        return True

    if intent.action == "chat":
        chat_text = intent.target or intent.raw
        related = memory_store.search(chat_text, 4)
        memory_lines = [item.content for item in related]
        _speak(brain.reply(chat_text, memory_lines), ui)

        auto_fact = _extract_memorable_fact(chat_text)
        if auto_fact:
            memory_store.add(auto_fact, "profile")
        return True

    if intent.requires_confirmation:
        _speak(f"Please confirm: {describe_for_confirmation(intent)}. Say yes or no.", ui)
        decision_known = False
        approved = False

        for _ in range(3):
            confirmation = ""
            if voice_mode and voice_listener is not None and voice_listener.available():
                heard = voice_listener.listen(
                    timeout_seconds=10.0,
                    phrase_time_limit_seconds=12.0,
                    tick=(lambda: ui.pump()) if ui is not None else None,
                )
                if not heard:
                    _speak("I did not catch yes or no. Please say yes or no.", ui)
                    continue
                confirmation = heard
                print(confirmation)
            else:
                try:
                    confirmation = input("Confirm> ")
                except EOFError:
                    return False

            if ui is not None and trim(confirmation):
                ui.add_message("You", confirmation)

            decision, decision_known = parse_confirmation(confirmation)
            if not decision_known:
                _speak("I did not catch yes or no. Please say yes or no.", ui)
                continue

            approved = decision
            break

        if not decision_known or not approved:
            _speak("Canceled.", ui)
            return True

    result = _execute_intent(intent, executor)
    _speak(result.message, ui)
    return intent.action != "exit"


def run(argv: list[str] | None = None) -> int:
    global _g_tts_enabled
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    voice_mode = to_lower(trim(args.mode)) == "voice"
    ui_enabled = args.ui if args.ui is not None else voice_mode

    config_path = args.config_path
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config: AliceConfig = load_config(config_path)
    _load_project_env(config.project_root)

    env_memory = os.getenv("ALICE_MEMORY_DB", "")
    if env_memory:
        memory_path = Path(env_memory)
        if memory_path.suffix == ".db":
            memory_path = memory_path.with_suffix(".tsv")
    else:
        memory_path = config.project_root / "data" / "alice_memory.tsv"

    memory_store = MemoryStore(memory_path)
    executor = AliceExecutor(config.allowed_roots, config.log_dir, config.max_runtime_seconds)
    brain = AliceBrain()
    voice_listener = VoiceListener() if voice_mode else None

    ui: Optional[AliceUI] = None
    if ui_enabled:
        candidate = AliceUI()
        if candidate.start():
            ui = candidate
            ui.set_state("idle")
            ui.set_status("Online")
            print("[Alice] UI: enabled")
        else:
            print("[Alice] UI: unavailable on this platform/runtime.")

    face_tracker: Optional[FaceTracker] = None
    if ui_enabled and args.camera:
        tracker = FaceTracker()
        if tracker.start(args.camera_index):
            print(f"[Alice] Camera tracking: enabled (camera {args.camera_index})")
            face_tracker = tracker
        else:
            camera_error = tracker.last_error()
            print(f"[Alice] Camera tracking: unavailable ({camera_error})")
            if ui is not None:
                ui.set_state("error")
                ui.set_status(f"Camera unavailable: {camera_error}")
                ui.pump()
    elif ui_enabled and not args.camera and ui is not None:
        ui.set_status("Camera disabled")

    _g_tts_enabled = (not args.no_tts) and _command_exists("say")
    if _g_tts_enabled:
        _configure_tts()

    print("[Alice] STT backend:", end=" ")
    if voice_mode:
        if voice_listener is not None and voice_listener.available():
            print(voice_listener.backend_name())
        else:
            reason = voice_listener.last_error() if voice_listener else "voice listener unavailable"
            print(f"none ({reason})")
            print("[Alice] Voice mode unavailable. Falling back to text input.")
    else:
        print("disabled")

    if _g_tts_enabled:
        if _g_tts_voice_name:
            print(f"[Alice] TTS backend: say ({_g_tts_voice_name}, {_g_tts_rate_wpm} wpm)")
        else:
            print(f"[Alice] TTS backend: say ({_g_tts_rate_wpm} wpm)")
    else:
        print("[Alice] TTS backend: none")

    print(f"[Alice] LLM backend: {brain.llm_backend()}")
    if brain.using_llm():
        _speak(f"Alice is online with conversational mode enabled using {brain.llm_backend()}.", ui)
    else:
        _speak("Alice is online. Advanced AI chat is unavailable, so I will use built-in responses.", ui)
    print("[Alice] NLU mode: rule-based intent parsing")
    print(f"[Alice] Memory items: {memory_store.count()} ({memory_store.db_path})")

    keep_running = True
    try:
        while keep_running:
            if ui is not None and face_tracker is not None:
                obs = face_tracker.latest()
                ui.set_face_target(obs.x, obs.y, obs.found, obs.face_count)

            if ui is not None:
                ui.pump()
                if not ui.running():
                    print("[Alice] UI window closed. Exiting.")
                    break

            if args.command is not None:
                utterance = args.command
            else:
                if ui is not None:
                    ui.set_state("listening")
                    ui.set_status("Listening..." if not voice_mode else "Listening to microphone...")

                if voice_mode and voice_listener is not None and voice_listener.available():
                    heard = voice_listener.listen(
                        timeout_seconds=16.0,
                        phrase_time_limit_seconds=24.0,
                        tick=(lambda: ui.pump()) if ui is not None else None,
                    )
                    if not heard:
                        reason = trim(voice_listener.last_error())
                        if reason:
                            print(f"[no speech: {reason}]")
                        else:
                            print("[no speech]")
                        if args.once:
                            break
                        continue
                    utterance = heard
                    print(f"You (voice)> {utterance}")
                else:
                    try:
                        utterance = input("You> ")
                    except EOFError:
                        break

            if trim(utterance):
                if ui is not None:
                    ui.add_message("You", utterance)
                keep_running = _handle_utterance(
                    utterance,
                    args.wake_word,
                    args.require_wake,
                    executor,
                    brain,
                    memory_store,
                    voice_mode,
                    voice_listener,
                    ui,
                )

            if args.once or args.command is not None:
                break
    except KeyboardInterrupt:
        print("\n[Alice] Interrupted. Exiting.")

    executor.shutdown()
    _stop_tts()
    if face_tracker is not None:
        face_tracker.stop()
    if ui is not None:
        ui.set_state("offline")
        ui.set_status("Offline")
        ui.stop()

    return 0


def main() -> int:
    return run(sys.argv[1:])
