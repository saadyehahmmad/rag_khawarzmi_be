"""
Repository-root and data-directory paths shared by API, agent, ingestion, and health checks.

Centralizing VECTOR_STORE_PATH resolution avoids subtle mismatches when the value is relative.
Also owns shared screens.json I/O helpers so both ingestion and the agent use identical loading
and step-resolution logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Repo root (parent of the agent package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical path to the screens/flows definition file.
SCREENS_JSON_PATH: Path = PROJECT_ROOT / "docs" / "screens.json"


def vector_store_root() -> Path:
    """Resolved Chroma parent directory (same rules as ingestion.config / chroma_ingest)."""
    raw = Path(os.getenv("VECTOR_STORE_PATH", "./vector_stores"))
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def load_screens_json() -> dict:
    """
    Load SCREENS_JSON_PATH and return a normalized graph-format dict.

    Supports both the current graph format::

        {"images": {...}, "flows": [...]}

    and the legacy flat-list format (converted on-the-fly for backward compatibility).
    Returns an empty dict when the file is missing or malformed.
    """
    if not SCREENS_JSON_PATH.is_file():
        return {}
    try:
        data = json.loads(SCREENS_JSON_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "flows" in data:
            return data
        if isinstance(data, list):
            images: dict = {}
            flows: list = []
            for entry in data:
                steps = []
                for i, img in enumerate(entry.get("images", []), start=1):
                    img_id = img["file"].rsplit(".", 1)[0]
                    images[img_id] = {"file": img["file"], "caption": img.get("caption", "")}
                    steps.append({"step": i, "image": img_id})
                flows.append(
                    {
                        "id": entry.get("id", ""),
                        "system": entry.get("system", ""),
                        "keywords": entry.get("keywords", []),
                        "steps": steps,
                    }
                )
            return {"images": images, "flows": flows}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def resolve_flow_images(
    flow: dict,
    image_registry: dict[str, dict],
    language: str = "en",
) -> list[str]:
    """
    Resolve a single flow's steps to an ordered, deduplicated list of image filenames.

    Steps are sorted by their ``step`` key and each image ID is looked up in the
    central registry.  The same filename is never returned twice (a given image node
    can appear in multiple steps of the same flow).

    The ``file`` value in each image entry may be either a plain string (legacy) or
    a language-keyed dict ``{"en": "...", "ar": "..."}``.  When it is a dict the
    entry matching ``language`` is used, falling back to ``"en"`` and then to any
    available value so resolution never silently returns nothing.

    Args:
        flow: A single flow entry from screens.json.
        image_registry: The ``images`` dict from the top-level screens.json graph.
        language: The detected prompt language (``"ar"``, ``"en"``, or ``"mixed"``).
                  ``"mixed"`` resolves to ``"en"``.

    Returns:
        Ordered list of unique image filenames belonging to this flow.
    """
    # Treat "mixed" as English for image selection.
    lang = language if language in ("ar", "en") else "en"

    steps = sorted(
        enumerate(flow.get("steps", [])),
        key=lambda t: (t[1].get("step", 0), t[0]),
    )
    steps = [s for _, s in steps]
    seen: set[str] = set()
    result: list[str] = []
    for s in steps:
        img_id = s.get("image", "")
        node = image_registry.get(img_id)
        if not node:
            continue
        file_val = node.get("file", "")
        if isinstance(file_val, dict):
            # Language-keyed: pick requested lang → fallback to "en" → any value.
            fname = file_val.get(lang) or file_val.get("en") or next(iter(file_val.values()), "")
        else:
            fname = file_val
        if fname and fname not in seen:
            result.append(fname)
            seen.add(fname)
    return result
