"""Turn a judge's raw text into a validated verdict, or say exactly why it could not.

Ladder (stop at the first rung that yields a valid object):
  1. strict json.loads of the whole text
  2. strip markdown fences / prose, take the first balanced {...} object
  3. repair common breakage: trailing commas, single quotes, Python literals, smart quotes,
     unquoted keys, a truncated tail (close open strings/brackets)
  4. pydantic validation, with light coercion (score "4/5" -> 4, "Tie" -> "tie")
The caller re-asks the judge once with the error message if all rungs fail (judge.py);
after that the case is recorded as parse_failed. Never a silent default score.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ParseError(ValueError):
    pass


def _first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced: truncated output, let repair try to close it


def _close_truncated(s: str) -> str:
    stack, in_str, esc = [], False, False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    s = re.sub(r':\s*$', ': null', s)
    return s + "".join(reversed(stack))


def _repair(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", s)))
    s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'(\s*[:,}\]])", r'"\1"\2', s)  # 'x' -> "x" when it is a key/value token
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', s)  # unquoted keys
    s = _close_truncated(s)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    return s


def _coerce(obj):
    """Lenient fixes before validation: '4/5' or '4' -> 4; 'Tie'/'TIE' -> 'tie'; 'a' -> 'A'."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "score" and isinstance(v, str):
                m = re.match(r"\s*(\d+)", v)
                v = int(m.group(1)) if m else v
            if k == "score" and isinstance(v, float) and v.is_integer():
                v = int(v)
            if k == "winner" and isinstance(v, str):
                v = {"a": "A", "b": "B", "tie": "tie", "draw": "tie", "equal": "tie"}.get(v.strip().lower(), v)
            out[k] = _coerce(v)
        return out
    if isinstance(obj, list):
        return [_coerce(x) for x in obj]
    if isinstance(obj, str) and obj.strip().lower() in {"a", "b", "tie"} and len(obj.strip()) <= 3:
        return {"a": "A", "b": "B", "tie": "tie"}[obj.strip().lower()]
    return obj


def parse_json(text: str) -> tuple[dict, str]:
    """Returns (object, rung_used). Raises ParseError when nothing parses."""
    if not text or not text.strip():
        raise ParseError("empty response")
    try:
        return json.loads(text), "strict"
    except json.JSONDecodeError:
        pass
    body = _first_object(re.sub(r"```(?:json)?", "", text))
    if body is None:
        raise ParseError("no JSON object found in response")
    try:
        return json.loads(body), "extracted"
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair(body)), "repaired"
    except json.JSONDecodeError as e:
        raise ParseError(f"unrepairable JSON: {e}") from e


def parse_verdict(text: str, model: type[T]) -> tuple[T, str]:
    obj, rung = parse_json(text)
    try:
        return model.model_validate(_coerce(obj)), rung
    except ValidationError as e:
        raise ParseError(f"schema error: {e.errors()[:3]}") from e
