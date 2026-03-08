from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    action: str
    target: str | None = None
    pid: int | None = None
    requires_confirmation: bool = False
    raw: str = ""


def _strip_wake_phrase(text: str, wake_word: str) -> str | None:
    cleaned = text.strip()
    lowered = cleaned.lower()
    wake = wake_word.lower().strip()
    candidates = [wake, f"hey {wake}", f"ok {wake}", f"okay {wake}"]

    for candidate in candidates:
        if not lowered.startswith(candidate):
            continue
        boundary = len(candidate)
        if len(lowered) > boundary and lowered[boundary].isalnum():
            continue
        return cleaned[boundary:].lstrip(" ,:;-")
    return None


def _clean_target(target: str) -> str:
    target = target.strip().strip('"').strip("'")
    target = re.sub(r"\b(for me|please|right now|now)\b", "", target, flags=re.IGNORECASE)
    return target.strip(" .")


def parse_intent(text: str, wake_word: str = "alice", require_wake: bool = True) -> Intent | None:
    spoken = text.strip()
    if not spoken:
        return None

    command = spoken
    if require_wake:
        stripped = _strip_wake_phrase(spoken, wake_word=wake_word)
        if stripped is None:
            return None
        command = stripped

    command_lower = command.lower().strip()
    if not command_lower:
        return Intent(action="greet", raw=spoken)

    if command_lower in {"help", "what can you do", "commands"}:
        return Intent(action="help", raw=spoken)

    if command_lower in {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "who are you",
        "what is your name",
        "thanks",
        "thank you",
    }:
        return Intent(action="smalltalk", target=command_lower, raw=spoken)

    if command_lower in {"exit", "quit", "shutdown", "stop listening", "goodbye", "bye"}:
        return Intent(action="exit", raw=spoken)

    run_match = re.match(
        r"^(?:please\s+)?run\s+(?:this\s+)?(?:file\s+)?(?P<target>.+)$",
        command,
        flags=re.IGNORECASE,
    )
    if run_match:
        target = _clean_target(run_match.group("target"))
        return Intent(
            action="run_file",
            target=target or None,
            requires_confirmation=True,
            raw=spoken,
        )

    open_match = re.match(
        r"^(?:open|show)\s+(?:folder|directory)\s+(?P<target>.+)$",
        command,
        flags=re.IGNORECASE,
    )
    if open_match:
        target = _clean_target(open_match.group("target"))
        return Intent(action="open_folder", target=target or ".", raw=spoken)

    list_match = re.match(
        r"^(?:list|show)\s+(?:the\s+)?files(?:\s+in)?\s*(?P<target>.*)$",
        command,
        flags=re.IGNORECASE,
    )
    if list_match:
        target = _clean_target(list_match.group("target")) or "."
        return Intent(action="list_files", target=target, raw=spoken)

    stop_match = re.match(
        r"^(?:stop|kill|terminate)\s*(?:process)?(?:\s+(?P<pid>\d+))?$",
        command,
        flags=re.IGNORECASE,
    )
    if stop_match:
        pid_raw = stop_match.group("pid")
        pid = int(pid_raw) if pid_raw else None
        return Intent(
            action="stop_process",
            pid=pid,
            requires_confirmation=True,
            raw=spoken,
        )

    return Intent(action="chat", target=command.strip(), raw=spoken)
