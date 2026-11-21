"""Reading and appending JSON Lines files."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read_jsonl(path: str | Path) -> list[Record]:
    """All rows of `path`, or an empty list when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def append_jsonl(path: str | Path, rows: Iterable[Record]) -> None:
    """Append `rows` to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
