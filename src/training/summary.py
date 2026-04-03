from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def update_training_summary(
    summary_path: Path,
    model_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"models": {}}

    if "models" not in summary or not isinstance(summary["models"], dict):
        summary["models"] = {}

    summary["models"][model_key] = payload
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def read_training_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {"models": {}}
    return json.loads(summary_path.read_text())
