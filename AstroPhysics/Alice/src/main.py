from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from env_utils import load_project_env
from config import load_config
from brain import AliceBrain
from executor import AliceExecutor, ExecResult
from intent import Intent
from listener import BaseListener, ListenerError, TextListener, VoiceListener
from nlu import IntentRouter
from speaker import Speaker
from ui import AliceFaceUI


HELP_TEXT = (
    "Try commands like: Alice run <file>, Alice list files in <folder>, "
    "Alice open folder <folder>, Alice stop process, Alice exit."
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


def is_yes(text: str | None) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return normalized in {"y", "yes", "yeah", "yep", "confirm"}


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
    if intent.action == "exit":
        return ExecResult(True, "Shutting down.")
    return ExecResult(False, "I did not understand that command. Say 'Alice help'.")


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
    ui: AliceFaceUI | None,
) -> bool:
    if ui is not None:
        ui.set_state("thinking")
        ui.set_status("Understanding...")

    intent = router.parse(utterance, wake_word=wake_word, require_wake=require_wake)
    if intent is None:
        if ui is not None:
            ui.set_state("idle")
            ui.set_status("Online")
        return True

    if intent.action == "chat":
        _speak(speaker, brain.reply(intent.target or intent.raw), ui=ui)
        return True

    if intent.requires_confirmation:
        _speak(
            speaker,
            f"Please confirm: {describe_for_confirmation(intent)}. Say yes or no.",
            ui=ui,
        )
        confirmation = _run_blocking_with_ui(
            ui,
            lambda: listener.listen("Confirm> "),
            state="listening",
            status="Waiting for confirmation...",
        )
        if ui is not None and confirmation:
            ui.add_message("You", confirmation)
        if not is_yes(confirmation):
            _speak(speaker, "Canceled.", ui=ui)
            return True

    if ui is not None:
        ui.set_state("thinking")
        ui.set_status("Working...")
    result = execute_intent(intent, executor)
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
    parser.add_argument("--no-wake", action="store_true", help="Accept commands without wake word")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech replies")
    parser.add_argument("--ui", action="store_true", help="Show animated Alice face window")
    parser.add_argument("--once", action="store_true", help="Process a single command and exit")
    parser.add_argument("--command", default=None, help="Single command text (works with --once)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = CURRENT_DIR.parent

    load_project_env(project_root)

    config = load_config(args.config)
    listener = build_listener(args.mode)
    speaker = Speaker(enable_tts=not args.no_tts)
    print(f"[Alice] TTS backend: {speaker.backend_name}")
    ui: AliceFaceUI | None = None
    if args.ui:
        ui = AliceFaceUI(title="Alice Interface")
        ui.set_status("Booting...")

    brain = AliceBrain()
    router = IntentRouter()
    executor = AliceExecutor(
        allowed_roots=config.allowed_roots,
        log_dir=config.log_dir,
        max_runtime_seconds=config.max_runtime_seconds,
    )

    if brain.using_openai:
        _speak(speaker, "Alice is online with conversational mode enabled.", ui=ui)
    else:
        _speak(speaker, "Alice is online. Conversational mode is enabled.", ui=ui)
    if router.using_openai:
        print("[Alice] NLU mode: OpenAI intent parsing enabled")
    else:
        print("[Alice] NLU mode: fallback parser only")

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
                        require_wake=not args.no_wake,
                        listener=listener,
                        speaker=speaker,
                        executor=executor,
                        brain=brain,
                        router=router,
                        ui=ui,
                    )

                if args.once:
                    break
                if args.command is not None:
                    break
        except (KeyboardInterrupt, EOFError):
            print("\n[Alice] Shutdown requested. Exiting cleanly.")
    finally:
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
