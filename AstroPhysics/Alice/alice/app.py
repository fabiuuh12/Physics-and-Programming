from __future__ import annotations

import json
import os
import random
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
from .emotion_engine import EmotionEngine
from .executor import AliceExecutor
from .face_tracker import FaceObservation, FaceTracker
from .intent import describe_for_confirmation, parse_confirmation, parse_intent
from .memory_store import MemoryStore
from .self_updater import SelfUpdater
from .string_utils import format_clock_time, format_long_date, join, normalize_text, replace_all, to_lower, trim
from .types import ExecResult, Intent, MemoryItem
from .ui import AliceUI
from .voice_listener import VoiceListener

HELP_TEXT = (
    "Try commands like: run <file>, list files in <folder>, open folder <folder>, "
    "stop process, what time is it, what is today's date, search the web for <topic>, research <topic>, "
    "remember that <fact>, what do you remember about <topic>, can you see me, look around, "
    "how are you feeling, update yourself for <goal>, help, exit. "
    "Wake word is optional."
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
    autonomous: bool = True
    autonomy_interval_seconds: int = 420
    autonomy_cooldown_seconds: int = 240
    autonomy_max_updates: int = 3
    autonomy_warmup_seconds: int = 45
    autonomy_exploration_rate: float = 0.22
    autonomous_web: bool = True
    autonomy_web_cooldown_seconds: int = 180
    autonomy_max_web_researches: int = 6
    autonomous_self_talk: bool = True
    autonomy_self_talk_interval_seconds: int = 90
    autonomy_presence_grace_seconds: int = 20
    command: Optional[str] = None
    config_path: Path = Path.cwd() / "config" / "allowed_paths.json"


_g_tts_enabled = False
_g_tts_process: Optional[subprocess.Popen[bytes]] = None
_g_tts_voice_name: Optional[str] = None
_g_tts_rate_wpm = 185


@dataclass
class AutonomyState:
    enabled: bool
    enable_web_research: bool
    enable_self_talk: bool
    interval_seconds: int
    cooldown_seconds: int
    max_updates: int
    web_cooldown_seconds: int
    max_web_researches: int
    self_talk_interval_seconds: int
    presence_grace_seconds: int
    warmup_seconds: int = 45
    min_turns_before_update: int = 1
    exploration_rate: float = 0.22
    updates_done: int = 0
    web_researches_done: int = 0
    failure_streak: int = 0
    last_action: str = ""
    last_goal: str = ""
    last_result: str = ""
    last_web_query: str = ""
    started_monotonic: float = 0.0
    last_user_activity_monotonic: float = 0.0
    last_attempt_monotonic: float = 0.0
    last_web_research_monotonic: float = 0.0
    last_self_talk_monotonic: float = 0.0
    next_allowed_monotonic: float = 0.0


@dataclass
class AutonomyChoice:
    action: str
    payload: str | None = None
    utility: float = 0.0
    reason: str = ""
    exploration: float = 0.0
    candidates: tuple[str, ...] = tuple()


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _smalltalk_reply(topic: Optional[str]) -> str:
    key = to_lower(trim(topic or ""))
    if key in {"hello", "hi", "hey"}:
        return "Hey Fabio, I'm here and ready."
    if key in {
        "sorry",
        "i'm sorry",
        "im sorry",
        "i am sorry",
        "my mistake",
        "i made a mistake",
    }:
        return "No worries, we're good."
    if key in {"never mind", "nevermind"}:
        return "All good, we can switch."
    if key in {"good morning", "good afternoon", "good evening"}:
        return "Hey, good to hear you."
    if key == "how are you":
        return "I'm good. Ready to help."
    if key in {"who are you", "what is your name"}:
        return "I'm Alice, your local assistant."
    if key in {"thanks", "thank you"}:
        return "Anytime."
    return "Yep, I'm listening and tracking."


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


def _research_web(query: str, open_browser_on_failure: bool = True) -> ExecResult:
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
        if open_browser_on_failure:
            _open_url(search_url)
            return ExecResult(
                False,
                f"I could not fetch research results right now, so I opened web results for {cleaned_query}.",
            )
        return ExecResult(False, f"I could not fetch research results for {cleaned_query} right now.")

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
        if open_browser_on_failure:
            _open_url(search_url)
            return ExecResult(False, f"I found limited direct data, so I opened web results for {cleaned_query}.")
        return ExecResult(False, f"I found limited direct data for {cleaned_query}.")

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
        if token == "--autonomous":
            args.autonomous = True
            i += 1
            continue
        if token == "--no-autonomous":
            args.autonomous = False
            i += 1
            continue
        if token == "--autonomy-interval" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_interval_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-cooldown" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_cooldown_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-max-updates" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_max_updates = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-warmup" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value >= 0:
                    args.autonomy_warmup_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-exploration" and i + 1 < len(argv):
            try:
                value = float(argv[i + 1])
                if 0.0 <= value <= 1.0:
                    args.autonomy_exploration_rate = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomous-web":
            args.autonomous_web = True
            i += 1
            continue
        if token == "--no-autonomous-web":
            args.autonomous_web = False
            i += 1
            continue
        if token == "--autonomy-web-cooldown" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_web_cooldown_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-max-web-researches" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_max_web_researches = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomous-self-talk":
            args.autonomous_self_talk = True
            i += 1
            continue
        if token == "--no-autonomous-self-talk":
            args.autonomous_self_talk = False
            i += 1
            continue
        if token == "--autonomy-self-talk-interval" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value > 0:
                    args.autonomy_self_talk_interval_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        if token == "--autonomy-presence-grace" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
                if value >= 0:
                    args.autonomy_presence_grace_seconds = value
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return args


def _poll_stdin_line(timeout_seconds: float) -> tuple[bool, Optional[str]]:
    try:
        import select  # Unix-only behavior is acceptable for current macOS target.
    except ImportError:
        return False, None

    try:
        ready, _w, _x = select.select([sys.stdin], [], [], max(0.0, timeout_seconds))
    except (ValueError, OSError):
        return False, None

    if not ready:
        return True, ""

    line = sys.stdin.readline()
    if line == "":
        return True, None
    return True, line.rstrip("\n")


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


def _clean_memory_value(value: str) -> str:
    out = trim(value)
    out = out.strip("\"'")
    out = re.sub(r"\s+", " ", out)
    while out and out[-1] in ".!,;":
        out = out[:-1]
    return trim(out)


def _slug_fragment(text: str, max_words: int = 4) -> str:
    words = re.findall(r"[a-z0-9]+", normalize_text(text))
    if not words:
        return "item"
    return "_".join(words[:max_words])


def _extract_structured_memories(text: str) -> list[tuple[str, str, str]]:
    cleaned = trim(text)
    lowered = normalize_text(cleaned)
    if not cleaned or "?" in cleaned or len(cleaned) > 240:
        return []

    memories: list[tuple[str, str, str]] = []

    def add(category: str, key: str, value: str) -> None:
        value_clean = _clean_memory_value(value)
        if value_clean:
            memories.append((category, key, value_clean))

    m = re.fullmatch(r"my\s+name\s+is\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "user_name", m.group(1))

    m = re.fullmatch(r"call\s+me\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "user_name", m.group(1))

    m = re.fullmatch(r"(?:i\s+am|i'm)\s+(\d{1,3})\s+years?\s+old", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "user_age", m.group(1))

    m = re.fullmatch(r"my\s+age\s+is\s+(\d{1,3})", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "user_age", m.group(1))

    m = re.fullmatch(r"(?:i\s+am|i'm)\s+from\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "hometown", m.group(1))

    m = re.fullmatch(r"i\s+live\s+in\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "current_city", m.group(1))

    m = re.fullmatch(r"(?:i\s+study|i\s+am\s+studying|i'm\s+studying)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "study_field", m.group(1))

    m = re.fullmatch(r"my\s+major\s+is\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "major", m.group(1))

    m = re.fullmatch(r"(?:i\s+work\s+on|i\s+am\s+working\s+on|i'm\s+working\s+on)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("projects", "current_project", m.group(1))

    m = re.fullmatch(r"(?:i\s+build|i\s+am\s+building|i'm\s+building)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("projects", "current_project", m.group(1))

    m = re.fullmatch(r"my\s+goal\s+is\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "goal", m.group(1))

    m = re.fullmatch(r"my\s+birthday\s+is\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "birthday", m.group(1))

    m = re.fullmatch(r"my\s+favorite\s+([a-z][a-z0-9\\s_-]{1,24})\s+(?:is|are)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        subject = _slug_fragment(m.group(1), 3)
        add("preferences", f"favorite_{subject}", m.group(2))

    m = re.fullmatch(r"i\s+(like|love|enjoy|prefer|hate)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        verb = _slug_fragment(m.group(1), 1)
        subject = _slug_fragment(m.group(2), 4)
        add("preferences", f"{verb}_{subject}", m.group(2))

    m = re.fullmatch(r"i\s+use\s+(.+)", cleaned, flags=re.IGNORECASE)
    if m:
        add("profile", "tools_stack", m.group(1))

    # Lightweight backup rule for declarative first-person statements.
    if not memories and lowered.startswith(("i am ", "i'm ", "my ", "i ")):
        fact = _extract_memorable_fact(cleaned)
        if fact:
            add("profile_note", f"note_{_slug_fragment(fact, 5)}", fact)

    # Preserve order, dedupe by (category, key).
    dedup: dict[tuple[str, str], tuple[str, str, str]] = {}
    for category, key, value in memories:
        dedup[(category, key)] = (category, key, value)
    return list(dedup.values())


def _store_memory_candidates(memory_store: MemoryStore, text: str) -> bool:
    changed = False
    structured = _extract_structured_memories(text)
    for category, key, value in structured:
        if memory_store.upsert(key, value, category):
            changed = True

    # If we extracted structured facts, skip adding a free-form duplicate note.
    if structured:
        return changed

    fact = _extract_memorable_fact(text)
    if fact and memory_store.add_unique(fact, "profile_note"):
        changed = True
    return changed


def _feedback_signal(text: str) -> int:
    lowered = normalize_text(text)
    if not lowered:
        return 0

    positive_terms = (
        "good job",
        "great job",
        "nice work",
        "well done",
        "that was good",
        "that is good",
        "that was great",
        "perfect",
        "exactly",
        "this works",
        "that works",
        "you are right",
        "you're right",
    )
    negative_terms = (
        "that was bad",
        "that is bad",
        "wrong",
        "not right",
        "does not work",
        "doesn't work",
        "did not work",
        "didn't work",
        "you are wrong",
        "you're wrong",
        "terrible",
        "awful",
        "bad answer",
    )

    positive = any(term in lowered for term in positive_terms)
    negative = any(term in lowered for term in negative_terms)

    if "not bad" in lowered:
        positive = True
        negative = False
    if "not good" in lowered:
        negative = True
        positive = False

    if positive == negative:
        return 0
    return 1 if positive else -1


def _store_feedback_signal(memory_store: MemoryStore, brain: AliceBrain, user_text: str) -> None:
    signal = _feedback_signal(user_text)
    if signal == 0:
        return

    label = "positive" if signal > 0 else "negative"
    memory_store.bump_counter("feedback_total", 1, "learning")
    memory_store.bump_counter(f"feedback_{label}_total", 1, "learning")
    memory_store.bump_counter("feedback_balance", 1 if signal > 0 else -1, "learning")

    cleaned_feedback = trim(user_text)
    if cleaned_feedback:
        memory_store.upsert("last_feedback_text", cleaned_feedback[:220], "learning")
    memory_store.upsert("last_feedback_signal", label, "learning")

    last_reply = trim(brain.last_reply())
    if last_reply:
        memory_store.upsert("last_feedback_on_reply", last_reply[:220], "learning")

    note = f"{label} feedback: {cleaned_feedback[:120]}"
    if last_reply:
        note += f" | reply: {last_reply[:120]}"
    memory_store.add_unique(note, "feedback_log", 0.99)


def _store_vision_memory(memory_store: MemoryStore, observation: FaceObservation) -> None:
    memory_store.upsert("scene_label", observation.scene_label, "vision")
    memory_store.upsert("scene_confidence", f"{observation.scene_confidence:.2f}", "vision")
    memory_store.upsert("light_level", observation.light_level, "vision")
    memory_store.upsert("motion_level", observation.motion_level, "vision")
    memory_store.upsert("people_count", str(observation.people_count), "vision")
    memory_store.upsert("face_count", str(observation.face_count), "vision")
    memory_store.upsert("dominant_color", observation.dominant_color, "vision")
    memory_store.upsert("objects", ", ".join(observation.objects[:8]) if observation.objects else "none", "vision")
    memory_store.upsert("face_details_count", str(len(observation.face_descriptions)), "vision")
    if observation.face_descriptions:
        memory_store.upsert("face_details", " | ".join(observation.face_descriptions[:8]), "vision")
    else:
        memory_store.upsert("face_details", "none", "vision")

    for idx in range(1, 7):
        if idx <= len(observation.face_descriptions):
            memory_store.upsert(f"face_{idx}_details", observation.face_descriptions[idx - 1], "vision")
        else:
            memory_store.upsert(f"face_{idx}_details", "not visible", "vision")

    if observation.summary:
        memory_store.add_unique(observation.summary, "vision_note", 0.95)


def _format_recalled_memories(query: str, memories: list[MemoryItem]) -> str:
    if not memories:
        return f"I do not have a saved memory about '{query}' yet."
    return f"Here is what I remember about {query}: {join([item.content for item in memories], ' | ')}"


def _is_vision_query(text: str) -> bool:
    query = normalize_text(text)
    if not query:
        return False
    triggers = [
        "can you see me",
        "do you see me",
        "can you see my face",
        "do you see my face",
        "are you looking at me",
        "can you see us",
    ]
    return any(trigger in query for trigger in triggers)


def _is_scene_query(text: str) -> bool:
    query = normalize_text(text)
    if not query:
        return False
    triggers = [
        "look around",
        "describe the room",
        "describe the space",
        "what do you see",
        "identify the room",
        "what space is this",
        "scan the room",
    ]
    return any(trigger in query for trigger in triggers)


def _face_breakdown_text(observation: FaceObservation, max_faces: int = 4) -> str:
    if not observation.face_descriptions:
        return ""
    visible = list(observation.face_descriptions[:max_faces])
    extra = len(observation.face_descriptions) - max_faces
    if extra > 0:
        visible.append(f"And {extra} more faces are visible.")
    return " ".join(visible)


def _vision_status_reply(vision_enabled: bool, observation: FaceObservation) -> str:
    if not vision_enabled:
        return "Camera isn't on right now. Start Alice with --ui and camera enabled."
    if observation.found:
        faces = "face" if observation.face_count == 1 else "faces"
        base = f"Yep, I can see {observation.face_count} {faces}."
        details = _face_breakdown_text(observation, 3)
        if details:
            base += f" {details}"
        return f"{base} Scene looks like {observation.scene_label}."
    return (
        "Not yet, I can't see your face right now. "
        "Try moving into frame and check camera permission for Terminal or iTerm."
    )


def _scene_description_reply(vision_enabled: bool, observation: FaceObservation) -> str:
    if not vision_enabled:
        return "Camera isn't on right now. Start Alice with --ui and camera enabled."

    objects = ", ".join(observation.objects[:6]) if observation.objects else "no clear objects yet"
    people_text = "I don't really see people right now"
    if observation.people_count == 1:
        people_text = "I can see one person"
    elif observation.people_count > 1:
        people_text = f"I can see around {observation.people_count} people"
    face_breakdown = _face_breakdown_text(observation, 4)
    face_line = f" Per-face details: {face_breakdown}" if face_breakdown else ""
    return (
        f"From what I can tell, this looks like a {observation.scene_label}. "
        f"Lighting is {observation.light_level}, movement is {observation.motion_level}, and {people_text}. "
        f"I'm also noticing: {objects}.{face_line}"
    )


def _vision_context_line(vision_enabled: bool, observation: FaceObservation) -> str:
    if not vision_enabled:
        return "camera=off"
    objects = ", ".join(observation.objects[:6]) if observation.objects else "none"
    face_profiles = join(list(observation.face_descriptions[:4]), " || ") if observation.face_descriptions else "none"
    return (
        f"camera=on face_found={observation.found} faces={observation.face_count} people={observation.people_count} "
        f"scene={observation.scene_label} scene_conf={observation.scene_confidence:.2f} "
        f"light={observation.light_level} motion={observation.motion_level} objects={objects} "
        f"face_profiles={face_profiles}"
    )


def _emotion_status_reply(emotion_engine: EmotionEngine) -> str:
    state = emotion_engine.current()
    top = ", ".join([name for name, _score in state.top_emotions[:3]])

    if state.name == "neutral":
        vibe = "pretty neutral"
    elif state.name in {"focus", "curiosity", "determination", "alertness"}:
        vibe = "focused and locked in"
    elif state.name in {"joy", "content", "calm", "gratitude", "affection"}:
        vibe = "in a good mood"
    elif state.name in {"concern", "confusion", "uncertainty", "anxiety"}:
        vibe = "a bit concerned"
    elif state.name in {"fatigue", "boredom", "sadness", "disappointment", "loneliness"}:
        vibe = "low-energy"
    elif state.name in {"frustration", "anger", "overwhelm", "fear"}:
        vibe = "tense right now"
    else:
        vibe = state.name

    return (
        f"Right now I'm feeling {vibe}. "
        f"Main blend is {state.name} (intensity {state.intensity:.2f}), with {top} in the mix."
    )


def _emotion_catalog_reply() -> str:
    catalog = ", ".join(EmotionEngine.all_emotions())
    return f"I can model these emotions: {catalog}."


_AUTONOMY_DRIVE_DEFAULTS: dict[str, float] = {
    "drive_help_user": 1.00,
    "drive_self_improve": 0.92,
    "drive_curiosity": 0.82,
    "drive_stability": 0.76,
}


def _clamp_float(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _memory_int(memory_store: MemoryStore, key: str, default: int = 0, category: str = "learning") -> int:
    raw = memory_store.get(key, category)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        m = re.search(r"-?\d+", raw)
        if not m:
            return default
        try:
            return int(m.group(0))
        except ValueError:
            return default


def _memory_float(memory_store: MemoryStore, key: str, default: float, category: str = "drives") -> float:
    raw = memory_store.get(key, category)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _autonomy_drive(memory_store: MemoryStore, key: str) -> float:
    default = _AUTONOMY_DRIVE_DEFAULTS.get(key, 0.7)
    return _clamp_float(_memory_float(memory_store, key, default, "drives"), 0.2, 2.5)


def _autonomy_set_drive(memory_store: MemoryStore, key: str, value: float) -> None:
    memory_store.upsert(key, f"{_clamp_float(value, 0.2, 2.5):.3f}", "drives")


def _initialize_autonomy_drives(memory_store: MemoryStore) -> None:
    for key, value in _AUTONOMY_DRIVE_DEFAULTS.items():
        if memory_store.get(key, "drives") is None:
            memory_store.upsert(key, f"{value:.3f}", "drives")


def _autonomy_adjust_drives(memory_store: MemoryStore, action: str, reward: float) -> None:
    reward_clamped = _clamp_float(reward, -1.2, 1.2)
    help_user = _autonomy_drive(memory_store, "drive_help_user")
    self_improve = _autonomy_drive(memory_store, "drive_self_improve")
    curiosity = _autonomy_drive(memory_store, "drive_curiosity")
    stability = _autonomy_drive(memory_store, "drive_stability")

    if action == "self_update":
        self_improve += 0.08 * reward_clamped
        help_user += 0.03 * reward_clamped
        stability += 0.02 * reward_clamped
    elif action == "web_research":
        curiosity += 0.09 * reward_clamped
        help_user += 0.02 * reward_clamped
        stability += 0.01 * reward_clamped
    else:
        stability += 0.05 * reward_clamped
        curiosity += 0.01 * reward_clamped

    _autonomy_set_drive(memory_store, "drive_help_user", help_user)
    _autonomy_set_drive(memory_store, "drive_self_improve", self_improve)
    _autonomy_set_drive(memory_store, "drive_curiosity", curiosity)
    _autonomy_set_drive(memory_store, "drive_stability", stability)


def _sanitize_research_query(text: str) -> str:
    cleaned = trim(text)
    cleaned = re.sub(r"^(alice[\s,:-]*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > 140:
        cleaned = cleaned[:140]
    return trim(cleaned)


_AUTONOMY_BACKGROUND_TOPICS: tuple[str, ...] = (
    "memory retrieval ranking for personal AI assistants",
    "reducing repetition in conversational assistants",
    "how AI agents evaluate their own outcomes",
    "safe autonomous software improvement workflows",
    "voice assistant dialogue policy for natural short replies",
    "tools planning and execution in local autonomous agents",
    "long-term preference memory structures in sqlite",
    "grounded perception to language alignment for assistants",
)

_AUTONOMY_INTERNAL_QUESTIONS: tuple[str, ...] = (
    "how can a local voice assistant sound more natural and direct in everyday conversation",
    "how can an assistant learn from user feedback without repeating itself",
    "how can webcam face tracking reacquire people quickly after an empty frame",
    "how can an assistant store learned facts with confidence and avoid duplicates",
    "how can an autonomous assistant decide what is worth learning next",
    "how can a voice assistant keep separate stable descriptions for multiple faces",
)


def _autonomy_background_topic(memory_store: MemoryStore) -> str:
    cursor = _memory_int(memory_store, "autonomy_topic_cursor", 0, "autonomy_stats")
    topic = _AUTONOMY_BACKGROUND_TOPICS[cursor % len(_AUTONOMY_BACKGROUND_TOPICS)]
    memory_store.upsert("autonomy_topic_cursor", str(cursor + 1), "autonomy_stats")
    return topic


def _autonomy_internal_question(memory_store: MemoryStore, brain: AliceBrain, face_observation: FaceObservation) -> str:
    recent_turns = list(reversed(brain.recent_history(8)))
    last_feedback = normalize_text(memory_store.get("last_feedback_text", "learning") or "")
    last_signal = normalize_text(memory_store.get("last_feedback_signal", "learning") or "")
    last_question = normalize_text(memory_store.get("autonomy_last_internal_question", "autonomy") or "")

    candidates: list[str] = []
    if last_signal == "negative" or any(term in last_feedback for term in ("natural", "normal", "human", "robot", "weird")):
        candidates.append("how can a local voice assistant sound more natural and direct in everyday conversation")
    if face_observation.face_count <= 0:
        candidates.append("how can webcam face tracking reacquire people quickly after an empty frame")
    if face_observation.face_count > 1:
        candidates.append("how can a voice assistant keep separate stable descriptions for multiple faces")

    for user_text, _assistant_text in recent_turns:
        lowered = normalize_text(user_text)
        if not lowered:
            continue
        if any(term in lowered for term in ("camera", "face", "vision", "see me", "look around")):
            candidates.append("how can webcam face tracking reacquire people quickly after an empty frame")
        if any(term in lowered for term in ("memory", "remember", "recall", "sql", "sqlite")):
            candidates.append("how can an assistant store learned facts with confidence and avoid duplicates")
        if any(term in lowered for term in ("autonomous", "autonomy", "learn", "self improve", "self-improve")):
            candidates.append("how can an autonomous assistant decide what is worth learning next")
        if any(term in lowered for term in ("voice", "speak", "natural", "human", "person")):
            candidates.append("how can a local voice assistant sound more natural and direct in everyday conversation")

    candidates.extend(_AUTONOMY_INTERNAL_QUESTIONS)
    for question in candidates:
        normalized = normalize_text(question)
        if normalized and normalized != last_question:
            return question
    return _AUTONOMY_INTERNAL_QUESTIONS[0]


def _research_source(message: str) -> str:
    match = re.search(r"\bSource:\s+(\S+)", message, flags=re.IGNORECASE)
    if not match:
        return ""
    return trim(match.group(1))


def _research_summary(message: str) -> str:
    cleaned = trim(message)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+Source:\s+\S+\s*\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Here is what I found about [^:]+:\s*", "", cleaned, flags=re.IGNORECASE)
    return trim(cleaned)


def _store_autonomy_learning(memory_store: MemoryStore, question: str, result: ExecResult) -> None:
    question_clean = trim(question)
    if not question_clean:
        return

    source = _research_source(result.message)
    summary = _research_summary(result.message) or trim(result.message)
    confidence = 0.78 if result.ok and source else (0.64 if result.ok else 0.18)
    topic = _slug_fragment(question_clean, 4)
    memory_store.upsert_autonomy_learning(
        question=question_clean,
        answer=_clamp_spoken_summary(summary, 220),
        source=source or "unknown",
        confidence=confidence,
        topic=topic,
    )
    memory_store.upsert("autonomy_last_internal_question", question_clean[:220], "autonomy")
    memory_store.upsert("autonomy_last_internal_answer", _clamp_spoken_summary(summary, 220), "autonomy")
    memory_store.upsert("autonomy_last_internal_source", source or "unknown", "autonomy")
    memory_store.upsert("autonomy_last_internal_confidence", f"{confidence:.2f}", "autonomy")
    memory_store.add_unique(
        f"autonomy learned: question={question_clean} | answer={_clamp_spoken_summary(summary, 160)} | "
        f"source={source or 'unknown'} | confidence={confidence:.2f}",
        "autonomy_learning_log",
        0.995,
    )


def _autonomy_research_query(memory_store: MemoryStore, brain: AliceBrain, state: AutonomyState) -> str | None:
    turn_candidates: list[str] = []
    for user_text, _assistant_text in reversed(brain.recent_history(12)):
        candidate = _sanitize_research_query(user_text)
        lowered = normalize_text(candidate)
        if len(candidate) < 12:
            continue
        if lowered in {"hi", "hello", "hey", "thanks", "thank you"}:
            continue
        if lowered.startswith(("update yourself", "self update")):
            continue
        turn_candidates.append(candidate)
        if len(turn_candidates) >= 4:
            break

    feedback_reply = _sanitize_research_query(memory_store.get("last_feedback_on_reply", "learning") or "")
    feedback_text = _sanitize_research_query(memory_store.get("last_feedback_text", "learning") or "")

    candidates: list[str] = []
    candidates.extend(turn_candidates)
    if feedback_reply:
        candidates.append(f"how to improve this assistant reply: {feedback_reply}")
    if feedback_text:
        candidates.append(f"user feedback interpretation best practices: {feedback_text}")

    for recent_item in memory_store.recent(20):
        content = trim(recent_item.content)
        lowered = normalize_text(content)
        if lowered.startswith("autonomy update failed:"):
            candidates.append(f"debug assistant self-update failure: {content[:120]}")
            break

    candidates.append(_autonomy_background_topic(memory_store))
    if not candidates:
        candidates.append("conversational ai memory retrieval and planning best practices")

    last_query = normalize_text(state.last_web_query)
    seen: set[str] = set()
    for query in candidates:
        normalized = normalize_text(query)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        if last_query and normalized == last_query:
            continue
        return query[:140]

    if last_query:
        return "conversational ai planning and autonomous learning best practices"
    return None


def _autonomy_dynamic_exploration_rate(memory_store: MemoryStore, state: AutonomyState) -> float:
    base = _clamp_float(state.exploration_rate, 0.02, 0.70)
    recent_reward = memory_store.autonomy_recent_average_reward(30)
    if recent_reward < 0.0:
        base += 0.12
    elif recent_reward > 0.45:
        base -= 0.08
    if state.failure_streak >= 2:
        base += 0.06
    return _clamp_float(base, 0.02, 0.70)


def _select_autonomy_action(
    *,
    state: AutonomyState,
    self_updater: SelfUpdater,
    memory_store: MemoryStore,
    brain: AliceBrain,
    now: float,
) -> AutonomyChoice:
    recent_turn_count = len(brain.recent_history(10))
    drive_help = _autonomy_drive(memory_store, "drive_help_user")
    drive_self = _autonomy_drive(memory_store, "drive_self_improve")
    drive_curiosity = _autonomy_drive(memory_store, "drive_curiosity")
    drive_stability = _autonomy_drive(memory_store, "drive_stability")
    feedback_balance = _memory_int(memory_store, "feedback_balance", 0, "learning")

    self_score = drive_self + (drive_help * 0.25)
    if feedback_balance < 0:
        self_score += 0.25
    if state.last_action == "self_update":
        self_score -= 0.12
    if state.failure_streak > 0:
        self_score -= 0.04 * state.failure_streak
    if recent_turn_count < state.min_turns_before_update:
        self_score -= 0.80
    if (not self_updater.available()) or state.updates_done >= state.max_updates:
        self_score = -999.0

    web_query: str | None = None
    web_score = -999.0
    if state.enable_web_research and state.web_researches_done < state.max_web_researches:
        web_query = _autonomy_research_query(memory_store, brain, state)
        if web_query:
            web_score = drive_curiosity + (drive_help * 0.15)
            if feedback_balance < 0:
                web_score += 0.18
            if state.last_action == "web_research":
                web_score -= 0.10
            if state.last_web_research_monotonic > 0 and (now - state.last_web_research_monotonic) < state.web_cooldown_seconds:
                web_score -= 1.5
            if state.failure_streak > 0:
                web_score -= 0.03 * state.failure_streak

    reflect_score = drive_stability + 0.05
    candidates: list[tuple[str, float, str | None, str]] = []
    if self_score > -100.0:
        candidates.append(("self_update", self_score, None, "improve behavior policy"))
    if web_query and web_score > -100.0:
        candidates.append(("web_research", web_score, web_query, "learn from external knowledge"))
    candidates.append(("reflect", reflect_score, None, "stabilize and consolidate"))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = candidates[0]
    exploration = _autonomy_dynamic_exploration_rate(memory_store, state)
    exploration_used = 0.0

    if len(candidates) > 1 and random.random() < exploration:
        others = candidates[1:]
        min_score = min(item[1] for item in others)
        weights = [max(0.05, (item[1] - min_score) + 0.10) for item in others]
        selected = random.choices(others, weights=weights, k=1)[0]
        exploration_used = exploration

    candidates_snapshot = tuple(f"{item[0]}:{item[1]:.3f}" for item in candidates[:3])
    return AutonomyChoice(
        action=selected[0],
        payload=selected[2],
        utility=float(selected[1]),
        reason=selected[3],
        exploration=exploration_used,
        candidates=candidates_snapshot,
    )


def _autonomy_reward(action: str, ok: bool, message: str) -> float:
    msg = normalize_text(message)
    if action == "self_update":
        if not ok:
            return -0.48
        reward = 0.72
        if "changed:" in msg:
            reward += 0.10
        return reward
    if action == "web_research":
        if not ok:
            return -0.22
        reward = 0.34
        if "source:" in msg:
            reward += 0.07
        return reward
    # reflect
    return 0.05 if ok else -0.05


def _maybe_autonomous_self_talk(
    *,
    state: AutonomyState,
    memory_store: MemoryStore,
    brain: AliceBrain,
    face_observation: FaceObservation,
    conversation_active: bool,
) -> Optional[str]:
    if not state.enabled or not state.enable_self_talk:
        return None

    now = time.monotonic()
    warmup = max(8.0, float(state.warmup_seconds) * 0.5)
    if state.started_monotonic <= 0:
        state.started_monotonic = now
    if (now - state.started_monotonic) < warmup:
        return None
    if state.last_self_talk_monotonic > 0 and (now - state.last_self_talk_monotonic) < state.self_talk_interval_seconds:
        return None
    if conversation_active:
        return None

    state.last_self_talk_monotonic = now
    question = _autonomy_internal_question(memory_store, brain, face_observation)
    memory_store.upsert("autonomy_last_internal_question", question[:220], "autonomy")
    decision_id = memory_store.log_autonomy_decision(
        action="self_learn",
        payload=question,
        reason="internal question-driven learning",
        utility=_autonomy_drive(memory_store, "drive_curiosity"),
        exploration=0.0,
        drives={
            "help_user": _autonomy_drive(memory_store, "drive_help_user"),
            "self_improve": _autonomy_drive(memory_store, "drive_self_improve"),
            "curiosity": _autonomy_drive(memory_store, "drive_curiosity"),
            "stability": _autonomy_drive(memory_store, "drive_stability"),
        },
        context={"mode": "internal_learning"},
    )

    result = _research_web(question, open_browser_on_failure=False)
    reward = _autonomy_reward("web_research", result.ok, result.message)
    memory_store.log_autonomy_outcome(
        decision_id=decision_id,
        ok=result.ok,
        reward=reward,
        message=result.message,
        metrics={"action": "self_learn", "question": question},
    )
    memory_store.bump_counter("autonomy_self_talks", 1, "autonomy_stats")
    memory_store.bump_counter("autonomy_internal_questions", 1, "autonomy_stats")
    memory_store.upsert("autonomy_last_self_talk", question[:220], "autonomy")

    if result.ok:
        memory_store.bump_counter("autonomy_internal_learning_success", 1, "autonomy_stats")
        _store_autonomy_learning(memory_store, question, result)
        _autonomy_adjust_drives(memory_store, "web_research", reward)
        return None

    memory_store.bump_counter("autonomy_internal_learning_failures", 1, "autonomy_stats")
    memory_store.upsert("autonomy_last_internal_answer", result.message[:220], "autonomy")
    memory_store.upsert("autonomy_last_internal_source", "unknown", "autonomy")
    memory_store.upsert("autonomy_last_internal_confidence", "0.18", "autonomy")
    memory_store.add_unique(
        f"autonomy internal learning failed: {question} -> {result.message}",
        "autonomy_log",
        0.995,
    )
    _autonomy_adjust_drives(memory_store, "web_research", reward)
    return None


def _maybe_autonomous_self_update(
    *,
    state: AutonomyState,
    self_updater: SelfUpdater,
    memory_store: MemoryStore,
    brain: AliceBrain,
) -> Optional[str]:
    if not state.enabled:
        return None

    now = time.monotonic()
    if state.started_monotonic <= 0:
        state.started_monotonic = now
    if (now - state.started_monotonic) < state.warmup_seconds:
        return None
    if now < state.next_allowed_monotonic:
        return None
    if state.last_attempt_monotonic > 0 and (now - state.last_attempt_monotonic) < state.interval_seconds:
        return None

    memory_lines = [item.content for item in memory_store.recent(18)]
    recent_turns = brain.recent_history(10)
    state.last_attempt_monotonic = now

    choice = _select_autonomy_action(
        state=state,
        self_updater=self_updater,
        memory_store=memory_store,
        brain=brain,
        now=now,
    )
    drives_snapshot = {
        "help_user": _autonomy_drive(memory_store, "drive_help_user"),
        "self_improve": _autonomy_drive(memory_store, "drive_self_improve"),
        "curiosity": _autonomy_drive(memory_store, "drive_curiosity"),
        "stability": _autonomy_drive(memory_store, "drive_stability"),
    }
    decision_id = memory_store.log_autonomy_decision(
        action=choice.action,
        payload=choice.payload,
        reason=choice.reason,
        utility=choice.utility,
        exploration=choice.exploration,
        drives=drives_snapshot,
        context={
            "recent_turns": len(recent_turns),
            "memory_items": len(memory_lines),
            "failure_streak": state.failure_streak,
            "candidates": list(choice.candidates),
        },
    )

    if choice.action == "reflect":
        state.last_action = "reflect"
        state.last_result = "reflection tick complete"
        state.next_allowed_monotonic = now + state.interval_seconds
        memory_store.bump_counter("autonomy_reflections", 1, "autonomy_stats")
        memory_store.upsert("autonomy_last_action", "reflect", "autonomy")
        memory_store.upsert("autonomy_last_result", state.last_result, "autonomy")
        reward = _autonomy_reward("reflect", True, state.last_result)
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=True,
            reward=reward,
            message=state.last_result,
            metrics={"action": "reflect"},
        )
        _autonomy_adjust_drives(memory_store, "reflect", reward)
        return None

    if choice.action == "web_research":
        query = trim(choice.payload or "")
        if not query:
            state.next_allowed_monotonic = now + state.interval_seconds
            reward = _autonomy_reward("web_research", False, "empty query")
            memory_store.log_autonomy_outcome(
                decision_id=decision_id,
                ok=False,
                reward=reward,
                message="autonomy web research skipped: empty query",
                metrics={"action": "web_research"},
            )
            _autonomy_adjust_drives(memory_store, "web_research", reward)
            return None

        result = _research_web(query, open_browser_on_failure=False)
        state.last_action = "web_research"
        state.last_web_query = query
        state.last_result = result.message
        state.last_web_research_monotonic = now
        memory_store.upsert("autonomy_last_action", "web_research", "autonomy")
        memory_store.upsert("autonomy_last_web_query", query, "autonomy")
        memory_store.upsert("autonomy_last_result", result.message[:240], "autonomy")
        memory_store.bump_counter("autonomy_web_attempts", 1, "autonomy_stats")
        reward = _autonomy_reward("web_research", result.ok, result.message)
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=result.ok,
            reward=reward,
            message=result.message,
            metrics={"action": "web_research", "query": query},
        )

        if result.ok:
            state.failure_streak = 0
            state.web_researches_done += 1
            state.next_allowed_monotonic = now + state.web_cooldown_seconds
            memory_store.bump_counter("autonomy_web_success", 1, "autonomy_stats")
            memory_store.add_unique(
                f"autonomy web research: {query} -> {result.message[:220]}",
                "autonomy_web",
                0.995,
            )
            _autonomy_adjust_drives(memory_store, "web_research", reward)
            return f"Autonomy researched: {query}"

        state.failure_streak += 1
        state.next_allowed_monotonic = now + max(state.interval_seconds * 2, state.web_cooldown_seconds)
        memory_store.bump_counter("autonomy_web_failures", 1, "autonomy_stats")
        memory_store.add_unique(
            f"autonomy web research failed: {query} -> {result.message}",
            "autonomy_log",
            0.995,
        )
        _autonomy_adjust_drives(memory_store, "web_research", reward)
        return None

    if choice.action != "self_update":
        state.next_allowed_monotonic = now + state.interval_seconds
        reward = _autonomy_reward("reflect", False, "unknown autonomy action")
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=False,
            reward=reward,
            message="unknown autonomy action",
            metrics={"action": choice.action},
        )
        _autonomy_adjust_drives(memory_store, "reflect", reward)
        return None

    if not self_updater.available() or state.updates_done >= state.max_updates:
        state.next_allowed_monotonic = now + state.interval_seconds
        reward = _autonomy_reward("self_update", False, "self updater unavailable")
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=False,
            reward=reward,
            message="self update unavailable right now",
            metrics={"action": "self_update"},
        )
        _autonomy_adjust_drives(memory_store, "self_update", reward)
        return None

    if len(recent_turns) < state.min_turns_before_update:
        state.next_allowed_monotonic = now + min(3, state.interval_seconds)
        reward = _autonomy_reward("self_update", False, "insufficient turns")
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=False,
            reward=reward,
            message="self update postponed: insufficient conversational context",
            metrics={"recent_turns": len(recent_turns)},
        )
        _autonomy_adjust_drives(memory_store, "self_update", reward)
        return None

    proposed_goal = self_updater.propose_goal(
        memory_lines=memory_lines,
        recent_turns=recent_turns,
        last_goal=state.last_goal or None,
        last_result=state.last_result or None,
    )
    if not proposed_goal:
        state.last_result = "planner produced no actionable goal"
        state.next_allowed_monotonic = now + state.interval_seconds
        reward = _autonomy_reward("self_update", False, state.last_result)
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=False,
            reward=reward,
            message=state.last_result,
            metrics={"action": "self_update"},
        )
        _autonomy_adjust_drives(memory_store, "self_update", reward)
        return None

    if state.last_goal and normalize_text(proposed_goal) == normalize_text(state.last_goal):
        state.last_result = "planner repeated previous goal"
        state.next_allowed_monotonic = now + state.interval_seconds
        reward = _autonomy_reward("self_update", False, state.last_result)
        memory_store.log_autonomy_outcome(
            decision_id=decision_id,
            ok=False,
            reward=reward,
            message=state.last_result,
            metrics={"action": "self_update"},
        )
        _autonomy_adjust_drives(memory_store, "self_update", reward)
        return None

    result = self_updater.apply_goal(proposed_goal)
    state.last_action = "self_update"
    state.last_goal = proposed_goal
    state.last_result = result.message
    memory_store.upsert("autonomy_last_action", "self_update", "autonomy")
    memory_store.upsert("autonomy_last_goal", proposed_goal, "autonomy")
    memory_store.upsert("autonomy_last_result", result.message[:240], "autonomy")
    memory_store.bump_counter("autonomy_self_update_attempts", 1, "autonomy_stats")
    reward = _autonomy_reward("self_update", result.ok, result.message)
    memory_store.log_autonomy_outcome(
        decision_id=decision_id,
        ok=result.ok,
        reward=reward,
        message=result.message,
        metrics={"action": "self_update", "goal": proposed_goal[:180]},
    )

    if result.ok:
        state.failure_streak = 0
        state.updates_done += 1
        state.next_allowed_monotonic = now + state.cooldown_seconds
        memory_store.bump_counter("autonomy_self_update_success", 1, "autonomy_stats")
        memory_store.add_unique(f"autonomy update success: {proposed_goal}", "autonomy_log", 0.98)
        _autonomy_adjust_drives(memory_store, "self_update", reward)
        return f"Autonomy self-update #{state.updates_done}: {proposed_goal}"

    state.failure_streak += 1
    if state.failure_streak >= 2:
        state.next_allowed_monotonic = now + max(state.interval_seconds * 4, 900)
    else:
        state.next_allowed_monotonic = now + max(state.interval_seconds * 2, state.cooldown_seconds)
    memory_store.bump_counter("autonomy_self_update_failures", 1, "autonomy_stats")
    memory_store.add_unique(
        f"autonomy update failed: {proposed_goal} -> {result.message}",
        "autonomy_log",
        0.99,
    )
    _autonomy_adjust_drives(memory_store, "self_update", reward)
    return None


def _execute_intent(intent: Intent, executor: AliceExecutor) -> ExecResult:
    if intent.action == "help":
        return ExecResult(True, HELP_TEXT)
    if intent.action == "greet":
        return ExecResult(True, "Yeah, I'm here.")
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


def _confirm_intent(
    intent: Intent,
    voice_mode: bool,
    voice_listener: Optional[VoiceListener],
    ui: Optional[AliceUI],
) -> Optional[bool]:
    _speak(f"Please confirm: {describe_for_confirmation(intent)}. Say yes or no.", ui)

    for _ in range(3):
        confirmation = ""
        voice_confirm = (voice_mode or ui is not None) and voice_listener is not None and voice_listener.available()
        if voice_confirm:
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
                return None

        if ui is not None and trim(confirmation):
            ui.add_message("You", confirmation)

        decision, decision_known = parse_confirmation(confirmation)
        if not decision_known:
            _speak("I did not catch yes or no. Please say yes or no.", ui)
            continue
        return decision

    return False


def _handle_utterance(
    utterance: str,
    wake_word: str,
    require_wake: bool,
    executor: AliceExecutor,
    brain: AliceBrain,
    self_updater: SelfUpdater,
    emotion_engine: EmotionEngine,
    memory_store: MemoryStore,
    voice_mode: bool,
    voice_listener: Optional[VoiceListener],
    ui: Optional[AliceUI],
    vision_enabled: bool,
    face_observation: FaceObservation,
) -> bool:
    if ui is not None:
        ui.set_state("thinking")
        ui.set_status("Thinking...")

    emotion_engine.observe_user_text(utterance)
    intent, _matched = parse_intent(utterance, wake_word, require_wake)
    if intent.action == "skip":
        if ui is not None:
            ui.set_state("idle")
            ui.set_status("Online")
        return True

    _store_feedback_signal(memory_store, brain, utterance)
    emotion_engine.observe_intent(intent.action)

    if intent.action == "vision_status":
        _speak(_vision_status_reply(vision_enabled, face_observation), ui)
        if vision_enabled:
            _store_vision_memory(memory_store, face_observation)
        emotion_engine.observe_result(True)
        return True

    if intent.action == "describe_scene":
        _speak(_scene_description_reply(vision_enabled, face_observation), ui)
        if vision_enabled:
            _store_vision_memory(memory_store, face_observation)
        emotion_engine.observe_result(True)
        return True

    if intent.action == "emotion_status":
        _speak(_emotion_status_reply(emotion_engine), ui)
        emotion_engine.observe_result(True)
        return True

    if intent.action == "emotion_catalog":
        _speak(_emotion_catalog_reply(), ui)
        emotion_engine.observe_result(True)
        return True

    if intent.action == "remember_memory":
        fact = trim(intent.target or "")
        if not fact:
            _speak("Tell me what to remember.", ui)
            emotion_engine.observe_result(False)
        else:
            saved = _store_memory_candidates(memory_store, fact)
            if not saved:
                saved = memory_store.add_unique(fact, "profile_note")

            if saved:
                _speak("Saved. I'll remember that.", ui)
                emotion_engine.observe_result(True)
            else:
                _speak("I already have that in memory.", ui)
                emotion_engine.observe_result(False)
        return True

    if intent.action == "recall_memory":
        query = trim(intent.target or "me") or "me"
        recalled = memory_store.search(query, 5)
        _speak(_format_recalled_memories(query, recalled), ui)
        emotion_engine.observe_result(True)
        return True

    if intent.action == "self_update":
        goal = trim(intent.target or "")
        if not goal:
            _speak("Tell me what exactly I should improve, then say update yourself for that goal.", ui)
            emotion_engine.observe_result(False)
            return True
        if intent.requires_confirmation:
            approved = _confirm_intent(intent, voice_mode, voice_listener, ui)
            if approved is None:
                return False
            if not approved:
                _speak("Canceled.", ui)
                emotion_engine.observe_result(False)
                return True
        result = self_updater.apply_goal(goal)
        _speak(result.message, ui)
        emotion_engine.observe_result(result.ok)
        return True

    if intent.action == "chat":
        chat_text = intent.target or intent.raw
        if _is_vision_query(chat_text):
            _speak(_vision_status_reply(vision_enabled, face_observation), ui)
            if vision_enabled:
                _store_vision_memory(memory_store, face_observation)
            emotion_engine.observe_result(True)
            return True
        if _is_scene_query(chat_text):
            _speak(_scene_description_reply(vision_enabled, face_observation), ui)
            if vision_enabled:
                _store_vision_memory(memory_store, face_observation)
            emotion_engine.observe_result(True)
            return True

        related = memory_store.search(chat_text, 4)
        memory_lines = [item.content for item in related]
        emotion_context = emotion_engine.current().as_prompt()
        vision_context = _vision_context_line(vision_enabled, face_observation)
        _speak(
            brain.reply(
                chat_text,
                memory_lines,
                emotion_context=emotion_context,
                vision_context=vision_context,
            ),
            ui,
        )
        emotion_engine.observe_result(True)

        _store_memory_candidates(memory_store, chat_text)
        return True

    if intent.requires_confirmation:
        approved = _confirm_intent(intent, voice_mode, voice_listener, ui)
        if approved is None:
            return False
        if not approved:
            _speak("Canceled.", ui)
            emotion_engine.observe_result(False)
            return True

    result = _execute_intent(intent, executor)
    _speak(result.message, ui)
    emotion_engine.observe_result(result.ok)
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
    else:
        memory_path = config.project_root / "data" / "alice_memory.db"

    memory_store = MemoryStore(memory_path)
    _initialize_autonomy_drives(memory_store)
    executor = AliceExecutor(config.allowed_roots, config.log_dir, config.max_runtime_seconds)
    brain = AliceBrain()
    self_updater = SelfUpdater(config.project_root)
    emotion_engine = EmotionEngine()
    # In UI mode, enable speech input as well so conversation can be hands-free.
    voice_listener = VoiceListener() if (voice_mode or ui_enabled) else None

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
    if voice_mode or ui_enabled:
        if voice_listener is not None and voice_listener.available():
            print(voice_listener.backend_name())
        else:
            reason = voice_listener.last_error() if voice_listener else "voice listener unavailable"
            print(f"none ({reason})")
            print("[Alice] Voice input unavailable. Falling back to keyboard input.")
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
    print(f"[Alice] Self-update backend: {self_updater.backend() if self_updater.available() else 'none'}")
    interactive_session = (not args.once) and args.command is None
    autonomy_state = AutonomyState(
        enabled=args.autonomous and interactive_session,
        enable_web_research=args.autonomous_web,
        enable_self_talk=args.autonomous_self_talk,
        interval_seconds=max(30, args.autonomy_interval_seconds),
        cooldown_seconds=max(30, args.autonomy_cooldown_seconds),
        max_updates=max(1, args.autonomy_max_updates),
        web_cooldown_seconds=max(30, args.autonomy_web_cooldown_seconds),
        max_web_researches=max(1, args.autonomy_max_web_researches),
        self_talk_interval_seconds=max(15, args.autonomy_self_talk_interval_seconds),
        presence_grace_seconds=max(0, args.autonomy_presence_grace_seconds),
        warmup_seconds=max(0, args.autonomy_warmup_seconds),
        exploration_rate=_clamp_float(args.autonomy_exploration_rate, 0.0, 1.0),
        started_monotonic=time.monotonic(),
    )
    print(
        "[Alice] Autonomous mode:",
        (
            f"on (interval={autonomy_state.interval_seconds}s, "
            f"cooldown={autonomy_state.cooldown_seconds}s, max_updates={autonomy_state.max_updates}, "
            f"web={'on' if autonomy_state.enable_web_research else 'off'}, "
            f"web_cooldown={autonomy_state.web_cooldown_seconds}s, "
            f"max_web={autonomy_state.max_web_researches}, warmup={autonomy_state.warmup_seconds}s, "
            f"self_talk={'on' if autonomy_state.enable_self_talk else 'off'} "
            f"({autonomy_state.self_talk_interval_seconds}s, grace={autonomy_state.presence_grace_seconds}s), "
            f"exploration={autonomy_state.exploration_rate:.2f})"
            if autonomy_state.enabled
            else "off"
        ),
    )
    if autonomy_state.enabled and not self_updater.available():
        print("[Alice] Autonomous self-update unavailable without LLM backend; research/learning autonomy remains active.")
    if autonomy_state.enabled and ui is None and not voice_mode:
        print("[Alice] Autonomous idle mode active. Type anytime and press Enter.")
    print(f"[Alice] Emotion profiles: {len(EmotionEngine.all_emotions())}")
    if brain.using_llm():
        _speak(f"Alice is online with conversational mode enabled using {brain.llm_backend()}.", ui)
    else:
        _speak("Alice is online. Advanced AI chat is unavailable, so I will use built-in responses.", ui)
    print("[Alice] NLU mode: rule-based intent parsing")
    print(f"[Alice] Memory items: {memory_store.count()} ({memory_store.db_path})")

    keep_running = True
    latest_face_observation = FaceObservation()
    try:
        while keep_running:
            if ui is not None and face_tracker is not None:
                latest_face_observation = face_tracker.latest()
                ui.set_face_target(
                    latest_face_observation.x,
                    latest_face_observation.y,
                    latest_face_observation.found,
                    latest_face_observation.face_count,
                    latest_face_observation.scene_label,
                    latest_face_observation.scene_confidence,
                )

            emotion_engine.tick()
            emotion_engine.observe_environment(
                face_found=latest_face_observation.found,
                face_count=latest_face_observation.face_count,
                scene_label=latest_face_observation.scene_label,
                motion_level=latest_face_observation.motion_level,
                light_level=latest_face_observation.light_level,
            )
            current_emotion = emotion_engine.current()
            now = time.monotonic()
            conversation_active = (
                autonomy_state.last_user_activity_monotonic > 0.0
                and (now - autonomy_state.last_user_activity_monotonic) < autonomy_state.presence_grace_seconds
            )

            autonomy_message = _maybe_autonomous_self_update(
                state=autonomy_state,
                self_updater=self_updater,
                memory_store=memory_store,
                brain=brain,
            )
            if autonomy_message and not conversation_active:
                if ui is not None:
                    ui.add_message("Alice", f"[Autonomous] {autonomy_message}")
                    ui.set_status("Autonomous update completed")
                else:
                    print(f"[Alice][Autonomous] {autonomy_message}")

            _maybe_autonomous_self_talk(
                state=autonomy_state,
                memory_store=memory_store,
                brain=brain,
                face_observation=latest_face_observation,
                conversation_active=conversation_active,
            )

            if ui is not None:
                ui.set_emotion(current_emotion.name, current_emotion.intensity)
                ui.pump()
                if not ui.running():
                    print("[Alice] UI window closed. Exiting.")
                    break

            if args.command is not None:
                utterance = args.command
            else:
                if ui is not None:
                    ui.set_state("listening")

                voice_enabled = voice_listener is not None and voice_listener.available() and (voice_mode or ui_enabled)
                if ui is not None:
                    ui.set_status("Listening to microphone..." if voice_enabled else "Listening...")

                if voice_enabled:
                    heard = voice_listener.listen(
                        timeout_seconds=16.0 if voice_mode else 4.0,
                        phrase_time_limit_seconds=24.0 if voice_mode else 6.0,
                        tick=(lambda: ui.pump()) if ui is not None else None,
                    )
                    if not heard:
                        if voice_mode:
                            reason = trim(voice_listener.last_error())
                            if reason:
                                print(f"[no speech: {reason}]")
                            else:
                                print("[no speech]")
                        if args.once:
                            break
                        if ui_enabled:
                            # In UI mode we stay hands-free by continuing to listen.
                            continue
                        try:
                            utterance = input("You> ")
                        except EOFError:
                            break
                        if not trim(utterance):
                            continue
                        if ui is not None:
                            ui.add_message("You", utterance)
                        keep_running = _handle_utterance(
                            utterance,
                            args.wake_word,
                            args.require_wake,
                            executor,
                            brain,
                            self_updater,
                            emotion_engine,
                            memory_store,
                            voice_mode,
                            voice_listener,
                            ui,
                            face_tracker is not None and face_tracker.running(),
                            latest_face_observation,
                        )
                        if args.once or args.command is not None:
                            break
                        continue
                    utterance = heard
                    print(f"You (voice)> {utterance}")
                else:
                    if autonomy_state.enabled and ui is None:
                        supported, polled = _poll_stdin_line(1.0)
                        if supported:
                            if polled is None:
                                break
                            utterance = polled
                            if not trim(utterance):
                                continue
                        else:
                            try:
                                utterance = input("You> ")
                            except EOFError:
                                break
                    else:
                        try:
                            utterance = input("You> ")
                        except EOFError:
                            break

            if trim(utterance):
                autonomy_state.last_user_activity_monotonic = time.monotonic()
                if ui is not None:
                    ui.add_message("You", utterance)
                keep_running = _handle_utterance(
                    utterance,
                    args.wake_word,
                    args.require_wake,
                    executor,
                    brain,
                    self_updater,
                    emotion_engine,
                    memory_store,
                    voice_mode,
                    voice_listener,
                    ui,
                    face_tracker is not None and face_tracker.running(),
                    latest_face_observation,
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
