from __future__ import annotations

import re
from typing import Optional

from .string_utils import normalize_text, to_lower, trim
from .types import Intent


_SMALLTALK = {
    "hello",
    "hi",
    "hey",
    "sorry",
    "i'm sorry",
    "im sorry",
    "i am sorry",
    "my mistake",
    "i made a mistake",
    "never mind",
    "nevermind",
    "it's okay",
    "its okay",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "who are you",
    "what is your name",
    "thanks",
    "thank you",
}


def _strip_wake_phrase(text: str, wake_word: str) -> Optional[str]:
    cleaned = trim(text)
    lowered = to_lower(cleaned)
    wake = to_lower(trim(wake_word))
    candidates = [wake, f"hey {wake}", f"ok {wake}", f"okay {wake}"]

    for candidate in candidates:
        if not lowered.startswith(candidate):
            continue
        if len(lowered) > len(candidate) and lowered[len(candidate)].isalnum():
            continue
        return trim(cleaned[len(candidate) :])
    return None


def _clean_target(raw: str) -> str:
    target = trim(raw)
    if target and target[0] in {'"', "'"}:
        target = target[1:]
    if target and target[-1] in {'"', "'", "."}:
        target = target[:-1]
    target = re.sub(r"\b(for me|please|right now|now)\b", "", target, flags=re.IGNORECASE)
    return trim(target)


def parse_intent(text: str, wake_word: str, require_wake: bool) -> tuple[Intent, bool]:
    spoken = trim(text)
    if not spoken:
        return Intent(action="chat", target="", raw=text), False

    command = spoken
    stripped = _strip_wake_phrase(spoken, wake_word)
    matched = False

    if require_wake and stripped is None:
        return Intent(action="skip", raw=spoken), False
    if stripped is not None:
        command = stripped
    matched = True

    lowered = to_lower(trim(command))
    if not lowered:
        return Intent(action="greet", raw=spoken), matched

    if lowered in {"help", "what can you do", "commands"}:
        return Intent(action="help", raw=spoken), matched

    m = re.fullmatch(
        r"(?:please\s+)?(?:remember|save|note|memorize|don't\s+forget|dont\s+forget)\s+(?:that\s+)?(.+)",
        command,
        flags=re.IGNORECASE,
    )
    if m:
        target = _clean_target(m.group(1))
        if target:
            return Intent(action="remember_memory", target=target, raw=spoken), matched

    m = re.fullmatch(
        r"(?:what\s+do\s+you\s+remember(?:\s+about)?|what\s+did\s+i\s+tell\s+you(?:\s+about)?|"
        r"what\s+have\s+you\s+learned(?:\s+about)?|what\s+do\s+you\s+know(?:\s+about)?|"
        r"recall|remember\s+about)\s*(.*)",
        command,
        flags=re.IGNORECASE,
    )
    if m:
        target = _clean_target(m.group(1)) or "me"
        return Intent(action="recall_memory", target=target, raw=spoken), matched

    if re.search(
        r"\b(what\s+time\s+is\s+it|what('?s| is)?\s+the\s+time|tell\s+me\s+the\s+time|"
        r"current\s+time|time\s+now|time\s+is\s+it|clock)\b",
        lowered,
    ):
        return Intent(action="get_time", raw=spoken), matched

    if re.search(
        r"\b(what('?s| is)?\s+the\s+date|what\s+day\s+is\s+it|what\s+is\s+today('?s)?\s+date|"
        r"today('?s| is)\s+date|today('?s| is)\s+day|current\s+date|date\s+today)\b",
        lowered,
    ):
        return Intent(action="get_date", raw=spoken), matched

    m = re.fullmatch(
        r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google)\s+(.+)",
        command,
        flags=re.IGNORECASE,
    )
    if m:
        target = _clean_target(m.group(1))
        if target:
            return Intent(action="web_search", target=target, raw=spoken), matched

    m = re.fullmatch(
        r"(?:research(?:\s+the\s+web)?(?:\s+for)?|do\s+research\s+on|investigate|"
        r"find\s+information\s+(?:on|about))\s+(.+)",
        command,
        flags=re.IGNORECASE,
    )
    if m:
        target = _clean_target(m.group(1))
        if target:
            return Intent(action="web_research", target=target, raw=spoken), matched

    m = re.fullmatch(
        r"(?:self[-\s]?update|update\s+yourself|improve\s+yourself|improve\s+your\s+code|"
        r"update\s+your\s+code|refactor\s+yourself)(?:\s+(?:for|to|so\s+that))?\s*(.*)",
        command,
        flags=re.IGNORECASE,
    )
    if m:
        target = _clean_target(m.group(1))
        return Intent(
            action="self_update",
            target=target if target else None,
            requires_confirmation=True,
            raw=spoken,
        ), matched

    if re.search(
        r"\b(can\s+you\s+see\s+me|do\s+you\s+see\s+me|can\s+you\s+see\s+my\s+face|do\s+you\s+see\s+my\s+face|are\s+you\s+looking\s+at\s+me|can\s+you\s+see\s+us)\b",
        lowered,
    ):
        return Intent(action="vision_status", raw=spoken), matched

    if re.search(
        r"\b(what\s+emotions\s+do\s+you\s+have|list\s+your\s+emotions)\b",
        lowered,
    ):
        return Intent(action="emotion_catalog", raw=spoken), matched

    if re.search(
        r"\b(how\s+are\s+you\s+feeling|what\s+do\s+you\s+feel)\b",
        lowered,
    ):
        return Intent(action="emotion_status", raw=spoken), matched

    if re.search(
        r"\b(look\s+around|scan\s+(the\s+)?(room|space|scene)|describe\s+(the\s+)?(room|space|scene)|what\s+do\s+you\s+see(\s+around)?|identify\s+(the\s+)?(room|space|scene)|what\s+space\s+is\s+this)\b",
        lowered,
    ):
        return Intent(action="describe_scene", raw=spoken), matched

    if lowered in _SMALLTALK:
        return Intent(action="smalltalk", target=lowered, raw=spoken), matched

    if lowered in {"exit", "quit", "shutdown", "stop listening", "goodbye", "bye"}:
        return Intent(action="exit", raw=spoken), matched

    m = re.fullmatch(r"(?:please\s+)?run\s+(?:this\s+)?(?:file\s+)?(.+)", command, flags=re.IGNORECASE)
    if m:
        target = _clean_target(m.group(1))
        return (
            Intent(
                action="run_file",
                target=target if target else None,
                requires_confirmation=True,
                raw=spoken,
            ),
            matched,
        )

    m = re.fullmatch(r"(?:open|show)\s+(?:folder|directory)\s+(.+)", command, flags=re.IGNORECASE)
    if m:
        target = _clean_target(m.group(1)) or "."
        return Intent(action="open_folder", target=target, raw=spoken), matched

    m = re.fullmatch(r"(?:list|show)\s+(?:the\s+)?files(?:\s+in)?\s*(.*)", command, flags=re.IGNORECASE)
    if m:
        target = _clean_target(m.group(1)) or "."
        return Intent(action="list_files", target=target, raw=spoken), matched

    m = re.fullmatch(r"(?:stop|kill|terminate)\s*(?:process)?(?:\s+(\d+))?", command, flags=re.IGNORECASE)
    if m:
        pid = int(m.group(1)) if m.group(1) else None
        return Intent(action="stop_process", pid=pid, requires_confirmation=True, raw=spoken), matched

    return Intent(action="chat", target=trim(command), raw=spoken), matched


def describe_for_confirmation(intent: Intent) -> str:
    if intent.action == "run_file":
        return f"run file '{intent.target or '(unknown)'}'"
    if intent.action == "stop_process":
        if intent.pid is not None:
            return f"stop process {intent.pid}"
        return "stop the latest process"
    if intent.action == "self_update":
        target = intent.target or "(no goal given)"
        return f"update Alice code for '{target}'"
    return intent.action


def parse_confirmation(text: str) -> tuple[bool, bool]:
    cleaned = normalize_text(text)
    if not cleaned:
        return False, False

    yes_terms = ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "go ahead"]
    no_terms = ["no", "nope", "nah", "cancel", "stop", "abort", "never mind", "nevermind"]

    has_yes = any(term in cleaned for term in yes_terms)
    has_no = any(term in cleaned for term in no_terms)

    if has_yes == has_no:
        return False, False
    return has_yes, True
