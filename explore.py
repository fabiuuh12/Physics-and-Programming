#!/usr/bin/env python3
"""Browse, inspect, and run the repository's programs using Python 3.10+."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'CompPhysics'
BUILD = PROJECT / 'build-launcher'


@dataclass(frozen=True)
class Entry:
    path: Path
    kind: str
    target: str = ''

    @property
    def name(self):
        return self.path.relative_to(PROJECT if self.path.is_relative_to(PROJECT) else ROOT).as_posix()

    @property
    def category(self):
        parts = Path(self.name).parts
        return '/'.join(parts[:2]) if parts[0] in {'CyberTools', 'simulations'} else parts[0] if len(parts) > 1 else 'general'

    @property
    def title(self):
        return self.path.stem.removesuffix('_viz').replace('_', ' ')


def catalog():
    entries = []
    cmake = (PROJECT / 'CMakeLists.txt').read_text()
    for target, source in re.findall(r'add_executable\(\s*(\S+)\s+"([^"]+)"\s*\)', cmake):
        entries.append(Entry(PROJECT / source, 'C++ demo', target))
    for path in [*PROJECT.rglob('*.py'), *(ROOT / 'CyberTools').rglob('*.py')]:
        parts = path.relative_to(ROOT).parts
        if any(p.startswith(('.', 'build')) or p in {'third_party', '__pycache__', 'tests'} for p in parts):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        executable = any(
            isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
            and any(isinstance(v, ast.Constant) and v.value == '__main__' for v in node.test.comparators)
            for node in tree.body
        )
        entries.append(Entry(path, 'Python script' if executable else 'Python module'))
    return sorted(entries, key=lambda e: e.name.lower())


def select(entries, query):
    normalized = query.replace('\\', '/').removeprefix('./').removeprefix('CompPhysics/')
    exact = [e for e in entries if normalized in {e.name, e.name.removeprefix('simulations/'), e.name.replace('simulations/defensive_systems/', 'DefensiveSys/'), e.path.name, e.path.stem, e.target, e.target.removesuffix('_cpp') if e.target else ''}]
    matches = exact or [e for e in entries if normalized.lower() in (e.name + ' ' + e.title).lower()]
    if len(matches) != 1:
        choices = '\n'.join('  ' + e.name for e in matches)
        raise ValueError(('Multiple matches; use a full path:\n' + choices) if matches else f'No match for {query!r}. Use the list command to browse.')
    return matches[0]


def display(entries):
    for i, entry in enumerate(entries, 1):
        print(f'{i:3}. [{entry.kind:13}] {entry.name}')


def describe(entry):
    print(f'{entry.title}\nType: {entry.kind}\nTopic: {entry.category}\nSource: {entry.path.relative_to(ROOT).as_posix()}')
    if entry.target:
        print(f'CMake target: {entry.target}\nRequires: CMake, a C++17 compiler, raylib, and a graphical desktop.')
    elif entry.kind == 'Python module':
        print('Supporting module: imported by other programs; no standalone entry point.')
    else:
        print('Uses the current Python environment. Some scripts require arguments or a webcam.\nRead the source or topic README for inputs and controls.')
    print('Run command: ' + subprocess.list2cmdline([sys.executable, 'explore.py', 'run', entry.name]))


def run_entry(entry, args, dry_run=False):
    if entry.kind == 'Python module':
        raise ValueError(f'{entry.name} is a supporting module, not a standalone program.')
    if entry.target:
        executable = BUILD / ('Release' if sys.platform == 'win32' else '') / (entry.target + ('.exe' if sys.platform == 'win32' else ''))
        commands = [
            ['cmake', '-S', str(PROJECT), '-B', str(BUILD), '-DCMAKE_BUILD_TYPE=Release'],
            ['cmake', '--build', str(BUILD), '--config', 'Release', '--target', entry.target],
            [str(executable), *args],
        ]
    else:
        commands = [[sys.executable, str(entry.path), *args]]
    for command in commands:
        print('+ ' + subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def doctor():
    print(f'Python: {sys.version.split()[0]} ({sys.executable})')
    for name in ('cmake', 'clang++', 'g++', 'cl'):
        print(f'{name}: {shutil.which(name) or "not found"}')
    print('\nC++ demos need CMake, one C++17 compiler, raylib and a desktop.\nPython packages: install CompPhysics/requirements.txt in a virtual environment.\nSee docs/GETTING_STARTED.md for setup and troubleshooting.\nThis check reports available commands; it does not verify raylib or camera access.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command')
    listing = subs.add_parser('list', help='Search programs and supporting modules')
    listing.add_argument('query', nargs='?', default='')
    subs.add_parser('doctor', help='Show environment and setup guidance')
    info = subs.add_parser('info', help='Explain a file and its requirements')
    info.add_argument('query')
    run = subs.add_parser('run', help='Build if needed, then launch one program')
    run.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    run.add_argument('query')
    run.add_argument('args', nargs=argparse.REMAINDER, help='Arguments forwarded to the program')
    options = parser.parse_args(argv)
    if options.command == 'doctor':
        doctor()
        return 0
    entries = catalog()
    try:
        if options.command == 'list':
            display([e for e in entries if options.query.lower() in (e.name + ' ' + e.title + ' ' + e.kind).lower()])
        elif options.command == 'info':
            describe(select(entries, options.query))
        elif options.command == 'run':
            args = options.args[1:] if options.args[:1] == ['--'] else options.args
            run_entry(select(entries, options.query), args, options.dry_run)
        else:
            runnable = [e for e in entries if e.kind != 'Python module']
            display(runnable)
            if not sys.stdin.isatty():
                print('\nUse: python3 explore.py run <name>')
                return 0
            answer = input('\nChoose a number (or q to quit): ').strip()
            if answer.lower() in {'q', ''}:
                return 0
            if not answer.isdecimal() or not 1 <= int(answer) <= len(runnable):
                raise ValueError('Choose a number from the displayed list.')
            entry = runnable[int(answer) - 1]
            describe(entry)
            run_entry(entry, [])
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        print('Setup help: docs/GETTING_STARTED.md', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
