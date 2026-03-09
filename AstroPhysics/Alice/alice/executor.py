from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .string_utils import join, normalize_text, replace_all, split_words, to_lower, trim
from .types import ExecResult


_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "third_party"}


@dataclass
class ProcessInfo:
    pid: int
    log_path: Path
    process: subprocess.Popen[bytes]


class AliceExecutor:
    def __init__(self, allowed_roots: list[Path], log_dir: Path, max_runtime_seconds: int = 300):
        self._allowed_roots = [self._normalize_path(root) for root in allowed_roots]
        self._log_dir = self._normalize_path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_runtime_seconds = max_runtime_seconds
        self._processes: dict[int, ProcessInfo] = {}
        self._last_pid: Optional[int] = None
        self._indexed_files: list[Path] = []

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    @staticmethod
    def _is_prefix_path(root: Path, value: Path) -> bool:
        try:
            value.relative_to(root)
            return True
        except ValueError:
            return root == value

    def _cleanup_finished(self) -> None:
        finished: list[int] = []
        for pid, info in self._processes.items():
            if info.process.poll() is not None:
                finished.append(pid)
        for pid in finished:
            self._processes.pop(pid, None)
            if self._last_pid == pid:
                self._last_pid = None

    def _is_allowed(self, path: Path) -> bool:
        normalized = self._normalize_path(path)
        return any(self._is_prefix_path(root, normalized) for root in self._allowed_roots)

    def _is_runnable_candidate(self, path: Path) -> bool:
        if not path.is_file():
            return False
        ext = to_lower(path.suffix)
        if ext in {".py", ".sh", ".bash"}:
            return True
        return os.access(path, os.X_OK)

    def _display_path(self, path: Path) -> str:
        normalized = self._normalize_path(path)
        for root in self._allowed_roots:
            if self._is_prefix_path(root, normalized):
                try:
                    rel = normalized.relative_to(root)
                except ValueError:
                    continue
                rel_text = rel.as_posix()
                return rel_text if rel_text else "."
        return str(normalized)

    def _resolve_target(
        self,
        raw_target: Optional[str],
        must_exist: bool,
        expect_directory: Optional[bool],
    ) -> tuple[Optional[Path], Optional[str]]:
        target = Path(trim(raw_target)) if raw_target and trim(raw_target) else Path(".")
        if not target.is_absolute():
            target = Path.cwd() / target
        target = self._normalize_path(target)

        if must_exist and not target.exists():
            return None, f"Path does not exist: {self._display_path(target)}"

        if expect_directory is not None and target.exists():
            if expect_directory and not target.is_dir():
                return None, f"Expected a folder but got: {self._display_path(target)}"
            if not expect_directory and not target.is_file():
                return None, f"Expected a file but got: {self._display_path(target)}"

        if not self._is_allowed(target):
            roots = [str(root) for root in self._allowed_roots]
            return None, f"Blocked by allowlist. Allowed roots: {join(roots, ', ')}"

        return target, None

    def _refresh_file_index(self) -> None:
        self._indexed_files = []
        for root in self._allowed_roots:
            if not root.exists():
                continue
            for current_root, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                base = Path(current_root)
                for name in files:
                    path = base / name
                    if self._is_runnable_candidate(path):
                        self._indexed_files.append(path)

    @staticmethod
    def _normalize_query(text: str) -> str:
        lowered = to_lower(trim(text))
        lowered = replace_all(lowered, "vizualization", "viz")
        lowered = replace_all(lowered, "visualization", "viz")
        lowered = replace_all(lowered, "visualisation", "viz")
        lowered = replace_all(lowered, "black hole", "blackhole")
        return normalize_text(lowered)

    @classmethod
    def _query_tokens(cls, text: str) -> list[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "file",
            "script",
            "program",
            "please",
            "for",
            "me",
            "open",
            "run",
            "execute",
            "start",
            "launch",
        }
        return [token for token in split_words(cls._normalize_query(text)) if token not in stop_words]

    def _score_candidate(self, query: str, candidate: Path) -> float:
        q_norm = self._normalize_query(query)
        tokens = self._query_tokens(query)
        if not q_norm and not tokens:
            return 0.0

        rel = str(candidate)
        for root in self._allowed_roots:
            if self._is_prefix_path(root, candidate):
                rel = str(candidate.relative_to(root))
                break

        rel_norm = self._normalize_query(rel)
        stem_norm = self._normalize_query(candidate.stem)

        def compact(value: str) -> str:
            return "".join(ch for ch in value if not ch.isspace())

        cq = compact(q_norm)
        cr = compact(rel_norm)
        cs = compact(stem_norm)

        score = 0.0
        if cq and cq in cr:
            score += 10.0
        if cq and cq in cs:
            score += 9.0

        for token in tokens:
            if token in cs:
                score += 3.0
            elif token in cr:
                score += 1.8

        score -= min(1.5, len(rel_norm) / 120.0)
        return score

    def _find_best_file_match(self, raw_target: str) -> tuple[Optional[Path], list[Path]]:
        self._refresh_file_index()
        if not self._indexed_files:
            return None, []

        scored = [(self._score_candidate(raw_target, path), path) for path in self._indexed_files]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        suggestions = [path for _, path in scored[:5]]
        if not scored or scored[0][0] < 3.6:
            return None, suggestions
        return scored[0][1], suggestions

    def _resolve_runnable_target(self, raw_target: Optional[str]) -> tuple[Optional[Path], Optional[str]]:
        path, error = self._resolve_target(raw_target, must_exist=False, expect_directory=False)
        if error:
            return None, error

        if path and path.exists() and path.is_file():
            if self._is_runnable_candidate(path):
                return path, None
            return None, f"File exists but is not directly runnable: {self._display_path(path)}"

        if not raw_target or not trim(raw_target):
            return None, "Missing file target."

        match, suggestions = self._find_best_file_match(raw_target)
        if match:
            return match, None

        if suggestions:
            names = [item.name for item in suggestions[:3]]
            return None, f"Could not find a runnable file for '{raw_target}'. Closest matches: {join(names, ', ')}"

        return None, f"Could not find a runnable file for '{raw_target}'."

    @staticmethod
    def _find_command(name: str) -> Optional[str]:
        return shutil.which(name)

    @staticmethod
    def _timestamp_for_filename() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _spawn_process(self, command: list[str], cwd: Path, log_path: Path) -> Optional[subprocess.Popen[bytes]]:
        if not command:
            return None
        try:
            log_file = open(log_path, "wb")
            process = subprocess.Popen(command, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
            return process
        except OSError:
            return None

    def list_files(self, target: Optional[str]) -> ExecResult:
        path, error = self._resolve_target(target, must_exist=True, expect_directory=True)
        if error:
            return ExecResult(False, error)
        assert path is not None

        entries = list(path.iterdir())
        entries.sort(key=lambda item: (not item.is_dir(), item.name.lower()))

        if not entries:
            return ExecResult(True, f"{self._display_path(path)} is empty.")

        lines = [self._display_path(path)]
        for entry in entries[:25]:
            lines.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if len(entries) > 25:
            lines.append(f"... ({len(entries) - 25} more)")

        return ExecResult(True, "\n".join(lines))

    def open_folder(self, target: Optional[str]) -> ExecResult:
        path, error = self._resolve_target(target, must_exist=True, expect_directory=True)
        if error:
            return ExecResult(False, error)
        assert path is not None
        return ExecResult(True, f"Folder ready: {self._display_path(path)}")

    def run_file(self, target: Optional[str]) -> ExecResult:
        self._cleanup_finished()

        path, error = self._resolve_runnable_target(target)
        if error:
            return ExecResult(False, error)
        if path is None:
            return ExecResult(False, "Missing runnable path.")

        stamp = self._timestamp_for_filename()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"run_{stamp}_{path.stem}.log"

        ext = to_lower(path.suffix)
        run_command: list[str]
        if ext == ".py":
            py = self._find_command("python3")
            if not py:
                return ExecResult(False, "python3 not found for .py file execution.")
            run_command = [py, str(path)]
        elif ext in {".sh", ".bash"}:
            bash = self._find_command("bash")
            if not bash:
                return ExecResult(False, "bash not found for shell script execution.")
            run_command = [bash, str(path)]
        else:
            if not os.access(path, os.X_OK):
                return ExecResult(False, "Unsupported file type. Use .py, .sh, .bash, or executable files.")
            run_command = [str(path)]

        process = self._spawn_process(run_command, path.parent, log_path)
        if not process:
            return ExecResult(False, "Failed to start process.")

        self._processes[process.pid] = ProcessInfo(pid=process.pid, log_path=log_path, process=process)
        self._last_pid = process.pid
        return ExecResult(True, f"Started {path.name} (pid {process.pid}). Logging to {log_path.name}.")

    def stop_process(self, pid: Optional[int]) -> ExecResult:
        self._cleanup_finished()

        target_pid = pid if pid is not None else self._last_pid
        if target_pid is None:
            return ExecResult(False, "No tracked running process to stop.")

        info = self._processes.get(target_pid)
        if info is None:
            return ExecResult(False, f"Process {target_pid} is not currently tracked.")

        process = info.process
        if process.poll() is not None:
            self._processes.pop(target_pid, None)
            if self._last_pid == target_pid:
                self._last_pid = None
            return ExecResult(True, f"Process {target_pid} already finished.")

        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

        self._processes.pop(target_pid, None)
        if self._last_pid == target_pid:
            self._last_pid = None

        return ExecResult(True, f"Stopped process {target_pid}.")

    def shutdown(self) -> None:
        for pid in list(self._processes.keys()):
            self.stop_process(pid)
