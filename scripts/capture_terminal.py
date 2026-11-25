"""Run real commands and save a terminal-style screenshot of their actual output.

    python scripts/capture_terminal.py docs/img/out.png "python -m rag.ingest data/corpus/raw --table demo" ...

Each command runs in a shell; the transcript ($ command + stdout/stderr) is recorded with Rich,
exported as SVG and rasterised with headless Chrome. Nothing is typed or edited by hand.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.text import Text

CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
NOISE = re.compile(r"(WARN lance|Fetching \d+ files|DeprecatedWarning|create_fts_index is deprecated|warnings\.warn)")


def main() -> None:
    out, cmds = Path(sys.argv[1]), sys.argv[2:]
    con = Console(record=True, width=118, force_terminal=True, color_system="truecolor")
    env = {**os.environ, "PYTHONUTF8": "1", "RUST_LOG": "error"}
    for cmd in cmds:
        con.print(Text("$ ", style="bold green") + Text(cmd, style="bold"))
        # shell=True on purpose: the commands are the operator's own argv (same trust as typing them into a
        # shell) and may use pipes. Never feed this script untrusted input.
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", env=env)
        lines = [ln for ln in (res.stdout + res.stderr).splitlines() if not NOISE.search(ln)]
        con.print(Text("\n".join(lines)), highlight=True)
    svg = con.export_svg(title=out.stem.replace("_", " "))
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w, h = (int(float(m.group(1))) + 4, int(float(m.group(2))) + 4) if m else (1400, 900)
    with tempfile.TemporaryDirectory() as td:
        html = Path(td) / "t.html"
        html.write_text(f"<html><body style='margin:0;background:#fff'>{svg}</body></html>", encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                        "--force-device-scale-factor=2", f"--screenshot={out.resolve()}", html.as_uri()], check=True,
                       capture_output=True)
    out.with_suffix(".svg").write_text(svg, encoding="utf-8")
    print(f"saved {out} ({w}x{h})")


if __name__ == "__main__":
    main()
