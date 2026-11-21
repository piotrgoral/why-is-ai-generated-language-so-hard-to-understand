"""Shapes of the rows stored in the JSONL files (the field names are the file format)."""

from typing import Literal, TypedDict

Setting = Literal["chat", "essay"]
SETTINGS: tuple[Setting, ...] = ("chat", "essay")


class Reply(TypedDict):
    """One assistant reply, a row of raw.jsonl."""

    setting: Setting
    model: str
    unit_id: str
    prompt: str
    turn: int
    text: str
    output_tokens: int
