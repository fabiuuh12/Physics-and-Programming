from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import load_config
from executor import AliceExecutor, ExecResult
from intent import Intent, parse_intent
from listener import BaseListener, ListenerError, TextListener, VoiceListener
from speaker import Speaker


HELP_TEXT = (
    "Try commands like: Alice run <file>, Alice list files in <folder>, "
    "Alice open folder <folder>, Alice stop process, Alice exit."
)


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


def handle_utterance(
    utterance: str,
    *,
    wake_word: str,
    require_wake: bool,
    listener: BaseListener,
    speaker: Speaker,
    executor: AliceExecutor,
) -> bool:
    intent = parse_intent(utterance, wake_word=wake_word, require_wake=require_wake)
    if intent is None:
        return True

    if intent.action == "unknown":
        speaker.say("I did not understand that command. Say 'Alice help'.")
        return True

    if intent.requires_confirmation:
        speaker.say(f"Please confirm: {describe_for_confirmation(intent)}. Say yes or no.")
        confirmation = listener.listen("Confirm> ")
        if not is_yes(confirmation):
            speaker.say("Canceled.")
            return True

    result = execute_intent(intent, executor)
    speaker.say(result.message)
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
    parser.add_argument("--once", action="store_true", help="Process a single command and exit")
    parser.add_argument("--command", default=None, help="Single command text (works with --once)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config)
    listener = build_listener(args.mode)
    speaker = Speaker(enable_tts=not args.no_tts)
    executor = AliceExecutor(
        allowed_roots=config.allowed_roots,
        log_dir=config.log_dir,
        max_runtime_seconds=config.max_runtime_seconds,
    )

    speaker.say("Alice is online. Say 'Alice help' for commands.")

    keep_running = True
    try:
        while keep_running:
            if args.command is not None:
                utterance = args.command
            else:
                utterance = listener.listen("You> ")

            if utterance:
                keep_running = handle_utterance(
                    utterance,
                    wake_word=args.wake_word,
                    require_wake=not args.no_wake,
                    listener=listener,
                    speaker=speaker,
                    executor=executor,
                )

            if args.once:
                break
            if args.command is not None:
                break
    finally:
        executor.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
