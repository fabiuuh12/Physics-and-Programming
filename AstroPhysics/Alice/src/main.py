from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from env_utils import load_project_env
from config import load_config
from brain import AliceBrain
from executor import AliceExecutor, ExecResult
from face_tracker import FaceTracker
from intent import Intent
from listener import BaseListener, ListenerError, TextListener, VoiceListener
from memory_store import MemoryStore
from nlu import IntentRouter
from speaker import Speaker
from ui import AliceFaceUI
from web_search import WebHit, WebSearcher


HELP_TEXT = (
    "Try commands like: run <file>, list files in <folder>, open folder <folder>, "
    "stop process, what time is it, what is today's date, remember that <fact>, "
    "what do you remember about <topic>, can you see my hand, exit. Wake word is optional."
)

T = TypeVar("T")


def smalltalk_reply(topic: str | None) -> str:
    key = (topic or "").strip().lower()
    if key in {"hello", "hi", "hey"}:
        return "Hello Fabio. I am ready."
    if key in {"good morning", "good afternoon", "good evening"}:
        return "Hello Fabio. Good to hear from you."
    if key == "how are you":
        return "I am doing well and ready to help with your project."
    if key in {"who are you", "what is your name"}:
        return "I am Alice, your local AI assistant."
    if key in {"thanks", "thank you"}:
        return "You are welcome."
    return "I am here and listening."


def build_listener(mode: str) -> BaseListener:
    if mode == "text":
        return TextListener()
    if mode == "voice":
        return VoiceListener()
    try:
        return VoiceListener()
    except ListenerError as exc:
        print(f"[Alice] Voice unavailable ({exc}). Falling back to text mode.")
        return TextListener()


def parse_confirmation(text: str | None) -> bool | None:
    if not text:
        return None

    lowered = text.strip().lower()
    cleaned = re.sub(r"[^\w\s']", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    if "yes or no" in cleaned or "say yes" in cleaned and " no" in f" {cleaned}":
        return None

    positive_phrases = {
        "go ahead",
        "do it",
        "run it",
        "open it",
        "sounds good",
        "confirm it",
        "yes please",
        "uh huh",
        "mhm",
        "mm hmm",
    }
    negative_phrases = {
        "do not",
        "don't",
        "dont",
        "no way",
        "not now",
        "cancel it",
        "stop that",
        "never mind",
        "nevermind",
        "uh uh",
    }

    has_positive_phrase = any(phrase in cleaned for phrase in positive_phrases)
    has_negative_phrase = any(phrase in cleaned for phrase in negative_phrases)

    tokens = set(re.findall(r"[a-z']+", cleaned))
    positive_tokens = {
        "y",
        "yes",
        "yeah",
        "yep",
        "yup",
        "yas",
        "yess",
        "yesss",
        "yees",
        "yeh",
        "guess",
        "jes",
        "ya",
        "sure",
        "ok",
        "okay",
        "affirmative",
        "confirm",
        "proceed",
        "run",
        "open",
        "doit",
    }
    negative_tokens = {"n", "no", "nope", "nah", "cancel", "stop", "abort"}

    positive_fuzzy = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "confirm", "proceed"}
    negative_fuzzy = {"no", "nope", "nah", "cancel", "stop", "abort"}

    def _has_close_match(source: set[str], targets: set[str], cutoff: float) -> bool:
        for token in source:
            if difflib.get_close_matches(token, list(targets), n=1, cutoff=cutoff):
                return True
        return False

    has_positive = (
        has_positive_phrase
        or bool(tokens & positive_tokens)
        or _has_close_match(tokens, positive_fuzzy, cutoff=0.78)
    )
    has_negative = (
        has_negative_phrase
        or bool(tokens & negative_tokens)
        or _has_close_match(tokens, negative_fuzzy, cutoff=0.82)
    )

    if has_positive and has_negative:
        return None
    if has_negative:
        return False
    if has_positive:
        return True
    return None


def describe_for_confirmation(intent: Intent) -> str:
    if intent.action == "run_file":
        return f"run file '{intent.target}'"
    if intent.action == "stop_process":
        return f"stop process {intent.pid}" if intent.pid else "stop the latest process"
    return intent.action


def execute_intent(intent: Intent, executor: AliceExecutor) -> ExecResult:
    if intent.action == "help":
        return ExecResult(True, HELP_TEXT)
    if intent.action == "greet":
        return ExecResult(True, "I am listening.")
    if intent.action == "smalltalk":
        return ExecResult(True, smalltalk_reply(intent.target))
    if intent.action == "list_files":
        return executor.list_files(intent.target)
    if intent.action == "open_folder":
        return executor.open_folder(intent.target)
    if intent.action == "run_file":
        return executor.run_file(intent.target)
    if intent.action == "stop_process":
        return executor.stop_process(intent.pid)
    if intent.action == "get_time":
        return ExecResult(True, f"It is {datetime.now().strftime('%I:%M %p')}.")
    if intent.action == "get_date":
        return ExecResult(True, f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.")
    if intent.action == "exit":
        return ExecResult(True, "Shutting down.")
    return ExecResult(False, "I did not understand that command. Say 'Alice help'.")


def _chat_context(face_tracker: FaceTracker | None) -> dict[str, str]:
    if face_tracker is None:
        return {"camera_enabled": "false"}

    obs = face_tracker.get_latest()
    return {
        "camera_enabled": "true",
        "camera_found_face": "true" if obs.found else "false",
        "camera_face_count": str(obs.face_count),
        "camera_found_hand": "true" if obs.hand_found else "false",
        "camera_hand_count": str(obs.hand_count),
        "camera_hand_backend": obs.hand_backend,
        "camera_owner_locked": "true" if obs.owner_locked else "false",
        "camera_owner_name": obs.owner_name or "unknown",
    }


def _looks_like_vision_question(text: str) -> bool:
    lowered = text.lower()
    vision_terms = ("see", "look", "watch", "camera", "track", "visible", "detect", "recognize")
    subject_terms = ("me", "my", "face", "hand", "hands", "fingers", "us", "this", "that", "you see")
    return any(term in lowered for term in vision_terms) and any(
        term in lowered for term in subject_terms
    )


def _camera_chat_reply(text: str, face_tracker: FaceTracker | None) -> str | None:
    lowered = text.strip().lower()
    if not lowered:
        return None

    if not _looks_like_vision_question(lowered):
        return None

    if face_tracker is None:
        return (
            "Camera tracking is not enabled right now. Start me with --ui --camera "
            "and I can look at you."
        )

    observation = face_tracker.get_latest()
    asks_hand = any(token in lowered for token in ("hand", "hands", "finger", "fingers", "palm"))
    asks_object = any(token in lowered for token in ("object", "item", "thing", "stuff"))

    if asks_object and not asks_hand:
        return (
            "I can detect faces and basic hand presence, but I do not have full object recognition yet."
        )

    if asks_hand:
        if observation.hand_found:
            if observation.hand_count == 1:
                return "Yes, I can see one hand."
            return f"Yes, I can see {observation.hand_count} hands."
        return (
            "I cannot see your hand clearly right now. Keep your hand inside the camera frame "
            "with brighter lighting."
        )

    if observation.found:
        name = observation.owner_name or "you"
        return f"Yes, I can see {name} and I am tracking your face."
    if observation.hand_found:
        return "I can see your hand, but I cannot lock your face right now."
    return (
        "I cannot see your face right now. Please face the camera and try better lighting."
    )


def _extract_memorable_fact(text: str) -> str | None:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None
    if "?" in cleaned:
        return None

    lowered = cleaned.lower()
    if len(cleaned) < 8 or len(cleaned) > 180:
        return None

    hedging = ("i think ", "i guess ", "probably ", "maybe ", "perhaps ")
    if any(lowered.startswith(prefix) for prefix in hedging):
        return None

    patterns = (
        r"^(?:my\s+(?:name|birthday|goal|project|major|school|city|hometown)\s+(?:is|are)\s+.+)$",
        r"^(?:my\s+favorite\s+[a-z][a-z\s]{1,20}\s+(?:is|are)\s+.+)$",
        r"^(?:i(?:'m| am)\s+working\s+on\s+.+)$",
        r"^(?:i\s+(?:study|work\s+on|build|use)\s+.+)$",
        r"^(?:i\s+(?:like|love|prefer|enjoy|hate)\s+.+)$",
        r"^(?:call\s+me\s+.+)$",
    )
    if not any(re.match(pattern, lowered) for pattern in patterns):
        return None

    cleaned = re.sub(r"[.!\s]+$", "", cleaned)
    return cleaned


def _format_recalled_memories(query: str, memories: list[str]) -> str:
    if not memories:
        return f"I do not have a saved memory about '{query}' yet."
    joined = " | ".join(memories)
    return f"Here is what I remember about {query}: {joined}"


def _web_fallback_reply(query: str, hits: list[WebHit]) -> str | None:
    if not hits:
        return None
    top = hits[0]
    source = top.source
    if top.url:
        source = f"{source}: {top.url}"
    return f"From web search: {top.snippet} (source {source})."


def _run_blocking_with_ui(
    ui: AliceFaceUI | None,
    fn: Callable[[], T],
    *,
    state: str | None = None,
    status: str | None = None,
) -> T:
    if ui is None:
        return fn()

    if state:
        ui.set_state(state)
    if status:
        ui.set_status(status)

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # pragma: no cover - propagated to caller
            error["exc"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    while worker.is_alive():
        ui.pump()
        if not ui.running:
            raise KeyboardInterrupt
        time.sleep(0.02)

    if "exc" in error:
        raise error["exc"]
    return result["value"]


def _speak(
    speaker: Speaker,
    text: str,
    *,
    ui: AliceFaceUI | None,
) -> None:
    if ui is not None:
        ui.add_message("Alice", text)
    _run_blocking_with_ui(
        ui,
        lambda: speaker.say(text),
        state="speaking",
        status="Speaking...",
    )
    if ui is not None:
        ui.set_state("idle")
        ui.set_status("Online")


def handle_utterance(
    utterance: str,
    *,
    wake_word: str,
    require_wake: bool,
    listener: BaseListener,
    speaker: Speaker,
    executor: AliceExecutor,
    brain: AliceBrain,
    router: IntentRouter,
    memory_store: MemoryStore,
    web_searcher: WebSearcher,
    ui: AliceFaceUI | None,
    face_tracker: FaceTracker | None,
) -> bool:
    if ui is not None:
        ui.set_state("thinking")
        ui.set_status("Understanding...")

    intent = _run_blocking_with_ui(
        ui,
        lambda: router.parse(utterance, wake_word=wake_word, require_wake=require_wake),
        state="thinking",
        status="Understanding...",
    )
    if intent is None:
        if ui is not None:
            ui.set_state("idle")
            ui.set_status("Online")
        return True

    if intent.action == "remember_memory":
        fact = (intent.target or "").strip()
        if not fact:
            _speak(speaker, "Tell me what to remember.", ui=ui)
            return True
        stored = memory_store.add(fact, category="profile")
        if stored:
            _speak(speaker, "Saved. I will remember that.", ui=ui)
        else:
            _speak(speaker, "I already remember that.", ui=ui)
        return True

    if intent.action == "recall_memory":
        query = (intent.target or "me").strip() or "me"
        recalled = _run_blocking_with_ui(
            ui,
            lambda: memory_store.search(query, limit=5),
            state="thinking",
            status="Thinking...",
        )
        reply = _format_recalled_memories(query, [item.content for item in recalled])
        _speak(speaker, reply, ui=ui)
        return True

    if intent.action == "chat":
        chat_text = intent.target or intent.raw
        context = _chat_context(face_tracker)
        related_memories = _run_blocking_with_ui(
            ui,
            lambda: memory_store.search(chat_text, limit=4),
            state="thinking",
            status="Thinking...",
        )
        memory_lines = [item.content for item in related_memories]
        web_hits: list[WebHit] = []
        web_lines: list[str] = []
        if web_searcher.should_search(chat_text):
            web_hits = _run_blocking_with_ui(
                ui,
                lambda: web_searcher.lookup(chat_text, max_results=3),
                state="thinking",
                status="Researching...",
            )
            web_lines = web_searcher.format_for_prompt(web_hits)

        if not brain.using_openai:
            camera_reply = _camera_chat_reply(chat_text, face_tracker)
            if camera_reply is not None:
                _speak(speaker, camera_reply, ui=ui)
                return True

        if brain.llm_backend == "none":
            web_reply = _web_fallback_reply(chat_text, web_hits)
            if web_reply is not None:
                _speak(speaker, web_reply, ui=ui)
            else:
                fallback_reply = _run_blocking_with_ui(
                    ui,
                    lambda: brain.reply(chat_text, context=context, memories=memory_lines),
                    state="thinking",
                    status="Thinking...",
                )
                _speak(speaker, fallback_reply, ui=ui)
        else:
            response_text = _run_blocking_with_ui(
                ui,
                lambda: brain.reply(
                    chat_text,
                    context=context,
                    memories=memory_lines,
                    web_facts=web_lines,
                ),
                state="thinking",
                status="Thinking...",
            )
            if web_hits and "source" not in response_text.lower():
                response_text = f"{response_text} Source: {web_hits[0].source}."
            _speak(
                speaker,
                response_text,
                ui=ui,
            )

        auto_fact = _extract_memorable_fact(chat_text)
        if auto_fact is not None:
            memory_store.add(auto_fact, category="profile")
        return True

    if intent.requires_confirmation:
        _speak(
            speaker,
            f"Please confirm: {describe_for_confirmation(intent)}. Say yes or no.",
            ui=ui,
        )
        max_attempts = 5 if isinstance(listener, VoiceListener) else 3
        attempts = 0
        while attempts < max_attempts:
            confirmation = _run_blocking_with_ui(
                ui,
                lambda: listener.listen("Confirm> "),
                state="listening",
                status="Waiting for confirmation...",
            )
            if ui is not None and confirmation:
                ui.add_message("You", confirmation)
            decision = parse_confirmation(confirmation)
            if decision is True:
                break
            if decision is False:
                _speak(speaker, "Canceled.", ui=ui)
                return True

            attempts += 1
            if attempts < max_attempts:
                _speak(speaker, "I did not catch yes or no. Please say yes or no.", ui=ui)
        else:
            _speak(speaker, "Canceled.", ui=ui)
            return True

    result = _run_blocking_with_ui(
        ui,
        lambda: execute_intent(intent, executor),
        state="thinking",
        status="Working...",
    )
    _speak(speaker, result.message, ui=ui)
    return intent.action != "exit"


def parse_args() -> argparse.Namespace:
    project_root = CURRENT_DIR.parent
    parser = argparse.ArgumentParser(description="Alice voice assistant (v1)")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "allowed_paths.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "text", "voice"],
        default="auto",
        help="Input mode",
    )
    parser.add_argument("--wake-word", default="Alice", help="Wake word")
    wake_group = parser.add_mutually_exclusive_group()
    wake_group.add_argument(
        "--require-wake",
        dest="require_wake",
        action="store_true",
        help="Require wake word before commands",
    )
    wake_group.add_argument(
        "--no-wake",
        dest="require_wake",
        action="store_false",
        help="Accept commands without wake word",
    )
    parser.set_defaults(require_wake=False)
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech replies")
    parser.add_argument("--ui", action="store_true", help="Show animated Alice face window")
    parser.add_argument("--camera", action="store_true", help="Enable camera face tracking (requires --ui)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for face tracking")
    parser.add_argument("--camera-owner", default="Fabio", help="Owner name shown when face lock is active")
    parser.add_argument("--once", action="store_true", help="Process a single command and exit")
    parser.add_argument("--command", default=None, help="Single command text (works with --once)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = CURRENT_DIR.parent

    load_project_env(project_root)

    config = load_config(args.config)
    listener = build_listener(args.mode)
    listener_backend = getattr(listener, "backend_name", None)
    if isinstance(listener_backend, str):
        print(f"[Alice] STT backend: {listener_backend}")
    speaker = Speaker(enable_tts=not args.no_tts)
    print(f"[Alice] TTS backend: {speaker.backend_name}")
    ui: AliceFaceUI | None = None
    if args.ui:
        ui = AliceFaceUI(title="Alice Interface")
        ui.set_status("Booting...")

    face_tracker: FaceTracker | None = None
    if args.camera and ui is None:
        print("[Alice] Camera tracking requires --ui. Ignoring --camera.")
    elif args.camera and ui is not None:
        face_tracker = FaceTracker(
            camera_index=args.camera_index,
            owner_name=args.camera_owner,
        )
        if face_tracker.start():
            ui.attach_face_tracker(face_tracker)
            print(
                f"[Alice] Camera tracking enabled (camera {args.camera_index}, owner {args.camera_owner})"
            )
        else:
            print(f"[Alice] Camera tracking unavailable: {face_tracker.last_error}")
            face_tracker = None

    brain = AliceBrain()
    router = IntentRouter()
    web_searcher = WebSearcher()
    memory_db_path = Path(
        os.getenv("ALICE_MEMORY_DB", str(project_root / "data" / "alice_memory.db"))
    ).expanduser()
    if not memory_db_path.is_absolute():
        memory_db_path = (project_root / memory_db_path).resolve()
    memory_store = MemoryStore(memory_db_path)
    executor = AliceExecutor(
        allowed_roots=config.allowed_roots,
        log_dir=config.log_dir,
        max_runtime_seconds=config.max_runtime_seconds,
    )

    print(f"[Alice] LLM backend: {brain.llm_backend}")

    if brain.llm_backend != "none":
        _speak(
            speaker,
            f"Alice is online with conversational mode enabled using {brain.llm_backend}.",
            ui=ui,
        )
    else:
        _speak(
            speaker,
            "Alice is online. Advanced AI chat is unavailable, so I will use built-in responses.",
            ui=ui,
        )
    if router.using_llm:
        print(f"[Alice] NLU mode: {router.llm_backend} intent parsing enabled")
    else:
        print("[Alice] NLU mode: fallback parser only")
    print(f"[Alice] Memory items: {memory_store.count()} ({memory_store.db_path})")
    print(f"[Alice] Web search mode: {web_searcher.mode}")

    keep_running = True
    try:
        try:
            while keep_running:
                if ui is not None:
                    ui.pump()
                    if not ui.running:
                        break

                if args.command is not None:
                    utterance = args.command
                else:
                    utterance = _run_blocking_with_ui(
                        ui,
                        lambda: listener.listen("You> "),
                        state="listening",
                        status="Listening...",
                    )

                if utterance:
                    if ui is not None:
                        ui.add_message("You", utterance)
                    keep_running = handle_utterance(
                        utterance,
                        wake_word=args.wake_word,
                        require_wake=args.require_wake,
                        listener=listener,
                        speaker=speaker,
                        executor=executor,
                        brain=brain,
                        router=router,
                        memory_store=memory_store,
                        web_searcher=web_searcher,
                        ui=ui,
                        face_tracker=face_tracker,
                    )

                if args.once:
                    break
                if args.command is not None:
                    break
        except (KeyboardInterrupt, EOFError):
            print("\n[Alice] Shutdown requested. Exiting cleanly.")
    finally:
        if face_tracker is not None:
            face_tracker.stop()
        if ui is not None:
            if ui.running:
                ui.set_state("offline")
                ui.set_status("Offline")
                ui.pump()
            ui.close()
        executor.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
