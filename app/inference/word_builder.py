"""Word composition placeholder.

The current runtime recognizes isolated static signs only: the letter camera
emits one of 24 static ASL classes and the digit camera emits ``0-9``. Neither
program composes predictions into words (see SYSTEM_CONTEXT.md, section 1).

This module reserves the canonical location for a future word-building stage.
It deliberately performs no processing yet and must not be imported by the
current inference path.
"""
from __future__ import annotations

__all__: list[str] = []
