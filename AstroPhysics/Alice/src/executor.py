from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    message: str


class AliceExecutor:
    def __init__(
        self,
        *,
        allowed_roots: list[Path],
        log_dir: Path,
        max_runtime_seconds: int = 300,
    ) -> None:
        self.allowed_roots = [path.resolve() for path in allowed_roots]
        self.log_dir = log_dir.resolve()
        self.max_runtime_seconds = max_runtime_seconds
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._processes: dict[int, tuple[subprocess.Popen[str], Path, object]] = {}
        self._last_pid: int | None = None

    def _cleanup_finished(self) -> None:
        finished: list[int] = []
        for pid, (process, _, log_handle) in self._processes.items():
            if process.poll() is not None:
                log_handle.close()
                finished.append(pid)
        for pid in finished:
            self._processes.pop(pid, None)
            if self._last_pid == pid:
                self._last_pid = None

    def _is_allowed(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.allowed_roots)

    def _resolve_target(
        self,
        raw_target: str | None,
        *,
        must_exist: bool = True,
        expect_directory: bool | None = None,
    ) -> tuple[Path | None, str | None]:
        if not raw_target:
            raw_target = "."

        target = Path(raw_target).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        else:
            target = target.resolve()

        if must_exist and not target.exists():
            return None, f"Path does not exist: {target}"

        if expect_directory is True and target.exists() and not target.is_dir():
            return None, f"Expected a folder but got: {target}"
        if expect_directory is False and target.exists() and not target.is_file():
            return None, f"Expected a file but got: {target}"

        if not self._is_allowed(target):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            return None, f"Blocked by allowlist. Allowed roots: {roots}"

        return target, None

    def list_files(self, target: str | None) -> ExecResult:
        path, error = self._resolve_target(target, expect_directory=True)
        if error:
            return ExecResult(False, error)

        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return ExecResult(True, f"{path} is empty.")

        preview = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries[:25]]
        suffix = "" if len(entries) <= 25 else f" ... ({len(entries) - 25} more)"
        message = f"{path}\n" + "\n".join(preview) + suffix
        return ExecResult(True, message)

    def open_folder(self, target: str | None) -> ExecResult:
        path, error = self._resolve_target(target, expect_directory=True)
        if error:
            return ExecResult(False, error)
        return ExecResult(True, f"Folder ready: {path}")

    def run_file(self, target: str | None) -> ExecResult:
        self._cleanup_finished()
        path, error = self._resolve_target(target, expect_directory=False)
        if error:
            return ExecResult(False, error)

        if path.suffix == ".py":
            command = [sys.executable, str(path)]
        elif path.suffix in {".sh", ".bash"}:
            command = ["bash", str(path)]
        elif os.access(path, os.X_OK):
            command = [str(path)]
        else:
            return ExecResult(
                False,
                "Unsupported file type. Use .py, .sh, .bash, or an executable file.",
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"run_{timestamp}_{path.stem}.log"
        log_handle = log_path.open("w", encoding="utf-8")

        try:
            process = subprocess.Popen(
                command,
                cwd=str(path.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            log_handle.close()
            return ExecResult(False, f"Failed to start process: {exc}")

        self._processes[process.pid] = (process, log_path, log_handle)
        self._last_pid = process.pid
        return ExecResult(
            True,
            f"Started {path.name} (pid {process.pid}). Output is being written to {log_path}.",
        )

    def stop_process(self, pid: int | None = None) -> ExecResult:
        self._cleanup_finished()

        target_pid = pid if pid is not None else self._last_pid
        if target_pid is None:
            return ExecResult(False, "No tracked running process to stop.")

        process_info = self._processes.get(target_pid)
        if process_info is None:
            return ExecResult(False, f"Process {target_pid} is not currently tracked.")

        process, _, log_handle = process_info
        if process.poll() is not None:
            log_handle.close()
            self._processes.pop(target_pid, None)
            if self._last_pid == target_pid:
                self._last_pid = None
            return ExecResult(True, f"Process {target_pid} already finished.")

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

        log_handle.close()
        self._processes.pop(target_pid, None)
        if self._last_pid == target_pid:
            self._last_pid = None

        return ExecResult(True, f"Stopped process {target_pid}.")

    def shutdown(self) -> None:
        for pid in list(self._processes.keys()):
            self.stop_process(pid)
