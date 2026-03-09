from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Optional

from .llm_client import ChatMessage, LLMClient
from .types import ExecResult


@dataclass(frozen=True)
class _PatchValidation:
    ok: bool
    message: str
    files: tuple[Path, ...] = ()


class SelfUpdater:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve(strict=False)
        self._llm = LLMClient()
        self._allowed_files = {
            Path("alice/app.py"),
            Path("alice/brain.py"),
            Path("alice/intent.py"),
            Path("alice/ui.py"),
            Path("alice/memory_store.py"),
            Path("alice/self_updater.py"),
            Path("README.md"),
        }
        self._max_changed_lines = 420

    def available(self) -> bool:
        return self._llm.available()

    def backend(self) -> str:
        return self._llm.backend()

    @staticmethod
    def _normalize_goal_line(raw: str) -> str:
        text = raw.strip()
        if not text:
            return ""
        text = text.splitlines()[0].strip()
        text = re.sub(r"^[-*0-9. )]+", "", text).strip()
        text = text.strip("\"'")
        return text

    @staticmethod
    def _goal_block_reason(goal: str) -> str | None:
        lowered = goal.lower()
        blocked_terms = (
            "disable safety",
            "remove safety",
            "remove guardrail",
            "bypass allowlist",
            "delete all",
            "rm -rf",
            "exfiltrate",
            "api key",
            "secret",
            "steal",
            "backdoor",
        )
        for term in blocked_terms:
            if term in lowered:
                return f"Goal blocked by safety rule: '{term}'."
        return None

    def propose_goal(
        self,
        *,
        memory_lines: list[str],
        recent_turns: list[tuple[str, str]],
        last_goal: str | None,
        last_result: str | None,
    ) -> str | None:
        if not self.available():
            return None

        memory_snippet = " ; ".join(memory_lines[:14]) if memory_lines else "none"
        turns = []
        for user_text, assistant_text in recent_turns[-8:]:
            turns.append(f"U: {user_text}\nA: {assistant_text}")
        turns_snippet = "\n".join(turns) if turns else "none"

        prompt = (
            "You are autonomous planner for Alice assistant. "
            "Return exactly one short self-improvement goal (single line, max 120 chars). "
            "The goal must target reply quality, autonomy, repetition reduction, memory quality, or planning quality. "
            "No markdown, no list, no explanation.\n\n"
            f"Last goal: {last_goal or 'none'}\n"
            f"Last result: {last_result or 'none'}\n"
            f"Recent memory: {memory_snippet}\n"
            f"Recent turns:\n{turns_snippet}\n"
        )
        messages = [
            ChatMessage(role="system", content="You output one concise software-improvement goal."),
            ChatMessage(role="user", content=prompt),
        ]
        raw = self._llm.chat(messages, temperature=0.2)
        goal = self._normalize_goal_line(raw)
        if not goal:
            return None
        if len(goal) > 120:
            goal = goal[:120].rstrip()
        if len(goal) < 8:
            return None
        if self._goal_block_reason(goal) is not None:
            return None
        return goal

    @staticmethod
    def _extract_diff(raw: str) -> str:
        text = raw.strip()
        if not text:
            return ""

        fenced = re.search(r"```diff\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        generic = re.search(r"```(?:patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if generic:
            return generic.group(1).strip()

        return text

    @staticmethod
    def _normalize_diff_path(raw: str) -> Optional[Path]:
        candidate = raw.strip()
        if not candidate or candidate == "/dev/null":
            return None
        if candidate.startswith("a/") or candidate.startswith("b/"):
            candidate = candidate[2:]
        path = Path(candidate)
        if path.is_absolute():
            return None
        return path

    def _validate_patch(self, diff_text: str) -> _PatchValidation:
        touched: list[Path] = []
        additions = 0
        deletions = 0

        for line in diff_text.splitlines():
            if line.startswith("+++ "):
                path = self._normalize_diff_path(line[4:])
                if path is None:
                    continue
                touched.append(path)
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        if not touched:
            return _PatchValidation(False, "No file changes found in proposed patch.")

        unique_files = tuple(dict.fromkeys(touched))
        for rel in unique_files:
            if rel not in self._allowed_files:
                return _PatchValidation(False, f"Patch tried to edit blocked file: {rel.as_posix()}")
            if ".." in rel.parts:
                return _PatchValidation(False, f"Patch path is invalid: {rel.as_posix()}")

        changed_lines = additions + deletions
        if changed_lines > self._max_changed_lines:
            return _PatchValidation(
                False,
                f"Patch too large ({changed_lines} changed lines). Limit is {self._max_changed_lines}.",
            )

        return _PatchValidation(True, "ok", unique_files)

    def _build_context(self, goal: str) -> str:
        chunks: list[str] = []
        for rel in sorted(self._allowed_files):
            abs_path = self._project_root / rel
            if not abs_path.exists() or not abs_path.is_file():
                continue
            try:
                text = abs_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if len(text) > 22000:
                text = text[:22000] + "\n# ...truncated..."
            chunks.append(f"FILE: {rel.as_posix()}\n{text}")
        combined = "\n\n".join(chunks)
        return (
            f"Goal:\n{goal}\n\n"
            "Constraints:\n"
            "- Return ONLY a unified diff.\n"
            "- Use paths relative to project root.\n"
            "- Edit only files listed in context.\n"
            "- Keep changes minimal and safe.\n"
            "- Preserve existing style and behavior unless needed for goal.\n\n"
            f"Project context:\n{combined}"
        )

    def _request_patch(self, goal: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a senior Python engineer generating code patches. "
                    "Output ONLY unified diff text. No prose."
                ),
            ),
            ChatMessage(role="user", content=self._build_context(goal)),
        ]
        raw = self._llm.chat(messages, temperature=0.15)
        return self._extract_diff(raw)

    def _backup_files(self, files: tuple[Path, ...]) -> dict[Path, Optional[bytes]]:
        backup: dict[Path, Optional[bytes]] = {}
        for rel in files:
            abs_path = self._project_root / rel
            if abs_path.exists():
                try:
                    backup[rel] = abs_path.read_bytes()
                except OSError:
                    backup[rel] = None
            else:
                backup[rel] = None
        return backup

    def _restore_files(self, backup: dict[Path, Optional[bytes]]) -> None:
        for rel, data in backup.items():
            abs_path = self._project_root / rel
            if data is None:
                if abs_path.exists():
                    try:
                        abs_path.unlink()
                    except OSError:
                        pass
                continue
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_bytes(data)
            except OSError:
                pass

    def _apply_diff(self, diff_text: str) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".diff", delete=False) as temp:
            temp.write(diff_text)
            patch_path = Path(temp.name)

        git_cmd = ["git", "apply", "--recount", "--whitespace=nowarn", str(patch_path)]
        try:
            proc = subprocess.run(
                git_cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            try:
                patch_path.unlink()
            except OSError:
                pass
            return False, f"Could not run git apply: {exc}"

        try:
            patch_path.unlink()
        except OSError:
            pass

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or "unknown patch apply failure"
            return False, f"Patch apply failed: {detail}"
        return True, "ok"

    def _compile_check(self) -> tuple[bool, str]:
        cmd = ["python3", "-m", "compileall", "alice"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Compile check failed to run: {exc}"

        if proc.returncode != 0:
            detail = (proc.stdout or "") + "\n" + (proc.stderr or "")
            detail = detail.strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "..."
            return False, f"Compile check failed: {detail}"
        return True, "ok"

    def apply_goal(self, goal: str) -> ExecResult:
        goal_clean = goal.strip()
        if not goal_clean:
            return ExecResult(False, "Tell me exactly what I should improve in my code.")
        if not self.available():
            return ExecResult(False, "Self-update needs an available LLM backend.")
        block_reason = self._goal_block_reason(goal_clean)
        if block_reason is not None:
            return ExecResult(False, block_reason)

        diff_text = self._request_patch(goal_clean)
        if not diff_text:
            return ExecResult(False, "I could not generate a patch for that goal.")

        validation = self._validate_patch(diff_text)
        if not validation.ok:
            return ExecResult(False, validation.message)

        backup = self._backup_files(validation.files)
        applied, apply_msg = self._apply_diff(diff_text)
        if not applied:
            return ExecResult(False, apply_msg)

        compiled, compile_msg = self._compile_check()
        if not compiled:
            self._restore_files(backup)
            return ExecResult(False, f"{compile_msg} I reverted the patch.")

        changed = ", ".join(path.as_posix() for path in validation.files)
        return ExecResult(True, f"Self-update completed. Changed: {changed}.")
