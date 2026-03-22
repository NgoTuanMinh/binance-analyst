"""Real-time monitor for batch download progress and disk usage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.utils import ensure_dir, get_file_size_str


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _disk_stats(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    used_pct = usage.used / usage.total * 100 if usage.total else 0
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "used_pct": round(used_pct, 2),
    }


def _eta_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    batches = checkpoint.get("batches", [])
    progress = checkpoint.get("progress", {})
    remaining = int(progress.get("remaining_batches", 0))
    if not batches or remaining <= 0:
        return "N/A"
    avg_seconds = sum(float(b.get("elapsed_seconds", 0)) for b in batches) / len(batches)
    eta_seconds = int(avg_seconds * remaining)
    return f"{eta_seconds // 3600}h {(eta_seconds % 3600) // 60}m {eta_seconds % 60}s"


def _render_html(checkpoint: dict[str, Any], disk: dict[str, Any], warning_pct: float) -> str:
    progress = checkpoint.get("progress", {})
    warning = disk["used_pct"] >= warning_pct
    last_batch = checkpoint.get("batches", [])[-1] if checkpoint.get("batches") else {}
    return f"""<html>
  <head><title>Batch Monitor Dashboard</title></head>
  <body>
    <h1>Batch Download Monitor</h1>
    <h2>Progress</h2>
    <ul>
      <li>Total batches: {progress.get("total_batches", 0)}</li>
      <li>Completed batches: {progress.get("completed_batches", 0)}</li>
      <li>Percent: {progress.get("percent", 0)}%</li>
      <li>Remaining batches: {progress.get("remaining_batches", 0)}</li>
      <li>ETA: {_eta_from_checkpoint(checkpoint)}</li>
    </ul>
    <h2>Disk Usage</h2>
    <ul>
      <li>Total: {get_file_size_str(disk["total"])}</li>
      <li>Used: {get_file_size_str(disk["used"])} ({disk["used_pct"]}%)</li>
      <li>Free: {get_file_size_str(disk["free"])}</li>
    </ul>
    <h3>Disk warning: {"YES" if warning else "NO"}</h3>
    <h2>Last Batch</h2>
    <pre>{json.dumps(last_batch, indent=2)}</pre>
    <h2>Generated At</h2>
    <p>{time.strftime("%Y-%m-%d %H:%M:%S")}</p>
  </body>
</html>
"""


def main() -> None:
    """CLI entrypoint for single-shot or watch-mode monitor."""
    parser = argparse.ArgumentParser(description="Monitor batch progress")
    parser.add_argument(
        "--checkpoint-file",
        default=str(PROJECT_ROOT / settings.LOG_DIR / "batch_download_checkpoint.json"),
    )
    parser.add_argument(
        "--output-html",
        default=str(PROJECT_ROOT / settings.LOG_DIR / "batch_monitor_dashboard.html"),
    )
    parser.add_argument("--warning-pct", type=float, default=85.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()

    checkpoint_file = Path(args.checkpoint_file)
    output_html = Path(args.output_html)
    ensure_dir(output_html.parent)

    def render_once() -> None:
        checkpoint = _read_checkpoint(checkpoint_file)
        disk = _disk_stats(PROJECT_ROOT)
        html = _render_html(checkpoint, disk, args.warning_pct)
        output_html.write_text(html, encoding="utf-8")
        print(
            f"Dashboard updated: {output_html} | "
            f"progress={checkpoint.get('progress', {}).get('percent', 0)}% | "
            f"disk_used={disk['used_pct']}%"
        )
        if disk["used_pct"] >= args.warning_pct:
            print("WARNING: disk usage is near full threshold.")

    if args.watch:
        while True:
            render_once()
            time.sleep(args.interval_seconds)
    else:
        render_once()


if __name__ == "__main__":
    main()
