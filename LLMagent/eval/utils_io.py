# eval/utils_io.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, List, Union, Optional
import numpy as np
import pandas as pd

JsonLike = Dict[str, Any]
PathLike = Union[str, Path]

def _json_default(o: Any):
    # json.dumps가 못하는 타입들 최소 처리
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    # numpy, torch 등은 여기서 확장 가능
    return str(o)

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_contexts(ctx):
    """
    contexts를 RAGAS가 기대하는 List[str]로 안전하게 변환.
    허용 입력:
      - None / missing
      - List[str]
      - List[{"text": "...", ...}]
    """
    if not ctx:
        return []
    if isinstance(ctx, list):
        if len(ctx) == 0:
            return []
        if isinstance(ctx[0], str):
            return [c for c in ctx if isinstance(c, str) and c.strip()]
        if isinstance(ctx[0], dict):
            out = []
            for c in ctx:
                if not isinstance(c, dict):
                    continue
                t = c.get("text", "")
                if isinstance(t, str) and t.strip():
                    out.append(t)
            return out
    return []