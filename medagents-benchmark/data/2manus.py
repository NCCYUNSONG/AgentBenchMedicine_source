#!/usr/bin/env python3
"""
Create slim versions of test_hard.jsonl that keep only
`realidx`, `question`, and `options`.

For every <dataset> directory under
/home/juanho-liang/PY/medagents-benchmark/data/
a file named test_hard_min.jsonl will be written.
"""

import json
from pathlib import Path

BASE_DIR = Path("/home/juanho-liang/PY/medagents-benchmark/data")
OUT_NAME = "test_hard_min.jsonl"           # output filename
KEEP = ("realidx", "question", "options")  # fields to preserve

def trim_file(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            trimmed = {k: entry[k] for k in KEEP if k in entry}
            fout.write(json.dumps(trimmed, ensure_ascii=False) + "\n")

def main() -> None:
    for dataset_dir in BASE_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        src = dataset_dir / "test_hard.jsonl"
        if not src.exists():
            continue
        dst = dataset_dir / OUT_NAME
        trim_file(src, dst)
        print(f"✔ Wrote {dst.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
