from __future__ import annotations

import difflib
import os
import re
import shlex
import shutil
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
    _RUNNABLE_EXTENSIONS = {".py", ".sh", ".bash", ".cpp", ".cc", ".cxx", ".c"}

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
        self._indexed_files: list[Path] = []

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

    def _is_runnable_candidate(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() in self._RUNNABLE_EXTENSIONS:
            return True
        return os.access(path, os.X_OK)

    def _refresh_file_index(self) -> None:
        skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "third_party"}
        candidates: list[Path] = []
        for root in self.allowed_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if any(part in skip_dirs for part in path.parts):
                    continue
                if self._is_runnable_candidate(path):
                    candidates.append(path)
        self._indexed_files = candidates

    def _normalize_query(self, text: str) -> str:
        lowered = text.strip().lower()
        replacements = {
            "vizualization": "viz",
            "visualization": "viz",
            "visualisation": "viz",
            "black hole": "blackhole",
        }
        for source, target in replacements.items():
            lowered = lowered.replace(source, target)
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _query_tokens(self, text: str) -> list[str]:
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
        normalized = self._normalize_query(text)
        return [tok for tok in normalized.split() if tok and tok not in stop_words]

    def _score_candidate(self, query: str, candidate: Path) -> float:
        query_norm = self._normalize_query(query)
        query_tokens = self._query_tokens(query)
        if not query_tokens and not query_norm:
            return 0.0

        rel = ""
        for root in self.allowed_roots:
            if candidate == root or root in candidate.parents:
                rel = str(candidate.relative_to(root))
                break
        if not rel:
            rel = str(candidate)

        rel_norm = self._normalize_query(rel)
        stem_norm = self._normalize_query(candidate.stem)

        score = 0.0
        compact_query = query_norm.replace(" ", "")
        compact_rel = rel_norm.replace(" ", "")
        compact_stem = stem_norm.replace(" ", "")

        if compact_query and compact_query in compact_rel:
            score += 10.0
        if compact_query and compact_query in compact_stem:
            score += 9.0

        for token in query_tokens:
            if token in compact_stem:
                score += 3.0
            elif token in compact_rel:
                score += 1.8

        if compact_query and compact_stem:
            score += 3.0 * difflib.SequenceMatcher(None, compact_query, compact_stem).ratio()
        if compact_query and compact_rel:
            score += 2.0 * difflib.SequenceMatcher(None, compact_query, compact_rel).ratio()

        # Prefer shorter/more direct matches among close candidates.
        score -= min(1.5, len(rel_norm) / 120.0)
        return score

    def _find_best_file_match(self, raw_target: str) -> tuple[Path | None, list[Path]]:
        self._refresh_file_index()
        if not self._indexed_files:
            return None, []

        scored = sorted(
            ((self._score_candidate(raw_target, path), path) for path in self._indexed_files),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scored:
            return None, []

        top_score, top_path = scored[0]
        if top_score < 3.6:
            return None, [path for _, path in scored[:5]]

        suggestions = [path for _, path in scored[:5]]
        return top_path, suggestions

    def _resolve_runnable_target(self, raw_target: str | None) -> tuple[Path | None, str | None]:
        path, error = self._resolve_target(
            raw_target,
            must_exist=False,
            expect_directory=False,
        )
        if path is not None and path.exists() and path.is_file():
            if self._is_runnable_candidate(path):
                return path, None
            return None, f"File exists but is not directly runnable: {path}"

        if not raw_target:
            return None, "Missing file target."

        matched, suggestions = self._find_best_file_match(raw_target)
        if matched is not None:
            return matched, None

        if suggestions:
            preview = ", ".join(path.name for path in suggestions[:3])
            return None, f"Could not find a runnable file for '{raw_target}'. Closest matches: {preview}"
        return None, f"Could not find a runnable file for '{raw_target}'."

    def _cxx_compiler(self) -> str | None:
        env_compiler = os.getenv("ALICE_CXX")
        if env_compiler:
            return env_compiler
        return shutil.which("clang++") or shutil.which("g++")

    def _cc_compiler(self) -> str | None:
        env_compiler = os.getenv("ALICE_CC")
        if env_compiler:
            return env_compiler
        return shutil.which("clang") or shutil.which("gcc")

    def _find_companion_executable(self, path: Path) -> Path | None:
        candidates = [
            path.with_suffix(""),
            path.parent / f"{path.stem}_cpp",
            path.parent / f"{path.stem}_bin",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _source_uses_raylib(self, path: Path) -> bool:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "raylib.h" in text

    def _raylib_pkg_config_flags(self) -> list[str]:
        if not shutil.which("pkg-config"):
            return []
        result = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "raylib"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        return shlex.split(result.stdout.strip())

    def _compile_extra_flags(self, path: Path, *, is_cpp: bool) -> list[str]:
        flags: list[str] = []
        if is_cpp:
            flags.extend(shlex.split(os.getenv("ALICE_CXXFLAGS", "").strip()))
        else:
            flags.extend(shlex.split(os.getenv("ALICE_CCFLAGS", "").strip()))
        flags.extend(shlex.split(os.getenv("ALICE_LDFLAGS", "").strip()))
        if self._source_uses_raylib(path):
            flags.extend(self._raylib_pkg_config_flags())
        return flags

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
        path, error = self._resolve_runnable_target(target)
        if error:
            return ExecResult(False, error)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"run_{timestamp}_{path.stem}.log"
        log_handle = log_path.open("w", encoding="utf-8")

        if path.suffix == ".py":
            command = [sys.executable, str(path)]
        elif path.suffix in {".sh", ".bash"}:
            command = ["bash", str(path)]
        elif path.suffix in {".cpp", ".cc", ".cxx"}:
            compiler = self._cxx_compiler()
            if not compiler:
                log_handle.close()
                return ExecResult(False, "C++ compiler not found. Install clang++ or g++.")
            bin_dir = self.log_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_path = bin_dir / f"{path.stem}_{timestamp}"
            compile_cmd = [
                compiler,
                str(path),
                "-std=c++17",
                "-O2",
                *self._compile_extra_flags(path, is_cpp=True),
                "-o",
                str(bin_path),
            ]
            log_handle.write("$ " + " ".join(compile_cmd) + "\n\n")
            log_handle.flush()
            compile_result = subprocess.run(
                compile_cmd,
                cwd=str(path.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if compile_result.returncode != 0:
                companion = self._find_companion_executable(path)
                if companion is not None:
                    command = [str(companion)]
                    log_handle.write(
                        f"\nCompilation failed for {path.name}. Falling back to {companion.name}.\n\n"
                    )
                    log_handle.flush()
                else:
                    log_handle.close()
                    return ExecResult(
                        False,
                        f"Compilation failed for {path.name}. See {log_path}.",
                    )
            else:
                command = [str(bin_path)]
        elif path.suffix == ".c":
            compiler = self._cc_compiler()
            if not compiler:
                log_handle.close()
                return ExecResult(False, "C compiler not found. Install clang or gcc.")
            bin_dir = self.log_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_path = bin_dir / f"{path.stem}_{timestamp}"
            compile_cmd = [
                compiler,
                str(path),
                "-O2",
                *self._compile_extra_flags(path, is_cpp=False),
                "-o",
                str(bin_path),
            ]
            log_handle.write("$ " + " ".join(compile_cmd) + "\n\n")
            log_handle.flush()
            compile_result = subprocess.run(
                compile_cmd,
                cwd=str(path.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if compile_result.returncode != 0:
                companion = self._find_companion_executable(path)
                if companion is not None:
                    command = [str(companion)]
                    log_handle.write(
                        f"\nCompilation failed for {path.name}. Falling back to {companion.name}.\n\n"
                    )
                    log_handle.flush()
                else:
                    log_handle.close()
                    return ExecResult(
                        False,
                        f"Compilation failed for {path.name}. See {log_path}.",
                    )
            else:
                command = [str(bin_path)]
        elif os.access(path, os.X_OK):
            command = [str(path)]
        else:
            log_handle.close()
            return ExecResult(
                False,
                "Unsupported file type. Use .py, .sh, .bash, .cpp, .cc, .cxx, .c, or an executable file.",
            )

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
