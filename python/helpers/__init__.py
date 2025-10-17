# helpers/__init__.py
"""
Small convenience package for session management.

Exports:
- SessionPaths: dataclass with resolved paths for a session
- init_session(project_root=".", session_name="Test 01") -> SessionPaths
- load_session(project_root=".", session_name=None) -> SessionPaths
- transforms_dir_for_session(session: SessionPaths) -> Path
- DIR_EXPORTED, DIR_INITIAL, DIR_MERGED: canonical subfolder names
"""

from .session_manager import (
    SessionPaths,
    init_session,
    load_session,
    transforms_dir_for_session,
    DIR_EXPORTED,
    DIR_INITIAL,
    DIR_MERGED,
)

__all__ = [
    "SessionPaths",
    "init_session",
    "load_session",
    "transforms_dir_for_session",
    "DIR_EXPORTED",
    "DIR_INITIAL",
    "DIR_MERGED",
]

__version__ = "0.1.0"
