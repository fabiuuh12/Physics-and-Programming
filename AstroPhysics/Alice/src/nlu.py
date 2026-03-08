from __future__ import annotations

import json
import os
import re

from intent import Intent, parse_intent


class IntentRouter:
    def __init__(self) -> None:
        self._client = None
        self._model = os.getenv("ALICE_INTENT_MODEL", os.getenv("ALICE_OPENAI_MODEL", "gpt-4o-mini"))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return

        try:
            from openai import OpenAI
        except ImportError:
            return

        self._client = OpenAI(api_key=api_key)

    @property
    def using_openai(self) -> bool:
        return self._client is not None

    def parse(self, text: str, *, wake_word: str, require_wake: bool) -> Intent | None:
        base_intent = parse_intent(text, wake_word=wake_word, require_wake=require_wake)
        if base_intent is None:
            return None

        if base_intent.action != "chat":
            return base_intent

        command_text = base_intent.target or base_intent.raw
        heuristic = self._heuristic_parse(command_text, raw=base_intent.raw)
        if heuristic is not None and heuristic.action != "chat":
            return heuristic

        llm_intent = self._parse_with_openai(command_text, raw=base_intent.raw)
        return llm_intent or base_intent

    def _clean_target(self, value: str) -> str:
        cleaned = value.strip().strip('"').strip("'")
        cleaned = re.sub(
            r"\b(please|for me|thanks|thank you|right now|now)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" .")

    def _extract_target_after_preposition(self, text: str) -> str | None:
        match = re.search(r"\b(?:in|inside|under|from)\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return None
        target = self._clean_target(match.group(1))
        return target or None

    def _heuristic_parse(self, command_text: str, *, raw: str) -> Intent | None:
        lowered = command_text.lower()

        if "what can you do" in lowered or lowered.strip() in {"help", "commands"}:
            return Intent(action="help", raw=raw)

        if any(word in lowered for word in {"quit", "exit", "shutdown", "stop listening"}):
            return Intent(action="exit", raw=raw)

        if any(word in lowered for word in {"stop", "kill", "terminate"}) and "process" in lowered:
            pid_match = re.search(r"\b(\d{2,8})\b", command_text)
            pid = int(pid_match.group(1)) if pid_match else None
            return Intent(action="stop_process", pid=pid, requires_confirmation=True, raw=raw)

        if any(word in lowered for word in {"list", "show", "display"}) and "file" in lowered:
            target = self._extract_target_after_preposition(command_text) or "."
            return Intent(action="list_files", target=target, raw=raw)
        if "what files" in lowered or "which files" in lowered:
            target = self._extract_target_after_preposition(command_text) or "."
            return Intent(action="list_files", target=target, raw=raw)

        if "open" in lowered and any(word in lowered for word in {"folder", "directory"}):
            target = self._extract_target_after_preposition(command_text)
            if target is None:
                match = re.search(
                    r"\b(?:open)\s+(?:the\s+)?(?:folder|directory)\s+(.+)$",
                    command_text,
                    flags=re.IGNORECASE,
                )
                if match:
                    target = self._clean_target(match.group(1))
            return Intent(action="open_folder", target=target or ".", raw=raw)

        run_match = re.search(
            r"\b(?:run|execute|start|launch)\b(?:\s+(?:the|this))?(?:\s+(?:file|script|program))?\s+(.+)$",
            command_text,
            flags=re.IGNORECASE,
        )
        if run_match:
            target = self._clean_target(run_match.group(1))
            if target:
                return Intent(
                    action="run_file",
                    target=target,
                    requires_confirmation=True,
                    raw=raw,
                )

        return Intent(action="chat", target=command_text, raw=raw)

    def _parse_with_openai(self, command_text: str, *, raw: str) -> Intent | None:
        if self._client is None:
            return None

        system_prompt = (
            "You classify a user's spoken request into one local assistant action.\n"
            "Return ONLY a JSON object with keys: action, target, pid.\n"
            "Valid actions: run_file, list_files, open_folder, stop_process, help, exit, chat.\n"
            "Rules:\n"
            "- If user wants to execute a script/program, choose run_file and include file path if present.\n"
            "- If user wants files shown, choose list_files.\n"
            "- If user wants a directory opened/checked, choose open_folder.\n"
            "- If user wants a process killed/stopped, choose stop_process and pid if present.\n"
            "- Choose help for capability questions.\n"
            "- Choose exit for shutdown/quit intents.\n"
            "- Otherwise choose chat.\n"
            "- Do not invent paths; if unknown keep target empty."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "could you show me what files are inside AstroPhysics/Alice please",
                    },
                    {"role": "assistant", "content": '{"action":"list_files","target":"AstroPhysics/Alice","pid":null}'},
                    {"role": "user", "content": "can you run examples/hello_alice.py for me"},
                    {"role": "assistant", "content": '{"action":"run_file","target":"examples/hello_alice.py","pid":null}'},
                    {"role": "user", "content": command_text},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
        except Exception:
            return None

        allowed_actions = {
            "run_file",
            "list_files",
            "open_folder",
            "stop_process",
            "help",
            "exit",
            "chat",
        }
        action = str(data.get("action", "chat")).strip().lower()
        if action not in allowed_actions:
            action = "chat"

        target_raw = data.get("target")
        target = str(target_raw).strip() if target_raw is not None else None
        if target == "":
            target = None

        pid = None
        pid_raw = data.get("pid")
        if pid_raw is not None:
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                pid = None

        if action in {"list_files", "open_folder"} and target is None:
            target = "."
        if action == "chat" and target is None:
            target = command_text

        return Intent(
            action=action,
            target=target,
            pid=pid,
            requires_confirmation=action in {"run_file", "stop_process"},
            raw=raw,
        )
