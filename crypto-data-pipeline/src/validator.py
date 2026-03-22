"""Validate merged datasets and generate quality reports/dashboard."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
import seaborn as sns

from src.utils import (
    ensure_dir,
    get_interval_minutes,
    normalize_open_time_ms_dataframe,
    progress_bar,
)


class DataValidator:
    """Dataset validator for merged symbol-level files."""

    def __init__(self, merged_dir: str):
        self.merged_dir = Path(merged_dir)
        self.report_dir = ensure_dir(self.merged_dir.parent.parent / "logs" / "reports")
        self._configure_plot_font()

    def _configure_plot_font(self) -> None:
        """Prefer CJK-capable fonts to avoid missing-glyph warnings on Windows."""
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Arial Unicode MS",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return

    def _safe_plot_symbol(self, symbol: Any) -> str:
        """Fallback display label when current font cannot render symbol chars."""
        text = str(symbol)
        try:
            text.encode("cp1252")
            return text
        except UnicodeEncodeError:
            return text.encode("ascii", "replace").decode("ascii")

    def check_data_continuity(self, df: pd.DataFrame, interval: str) -> list[dict[str, Any]]:
        """Find missing continuity gaps based on interval."""
        if df.empty:
            return []
        df = normalize_open_time_ms_dataframe(df)
        minutes = get_interval_minutes(interval)
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).sort_values().reset_index(drop=True)
        expected = pd.Timedelta(minutes=minutes)
        diffs = ts.diff()
        gap_points = diffs[diffs > expected]
        gaps: list[dict[str, Any]] = []
        for idx, diff in gap_points.items():
            prev_ts = ts.iloc[idx - 1] if idx > 0 else None
            gaps.append(
                {
                    "from": str(prev_ts) if prev_ts is not None else None,
                    "to": str(ts.iloc[idx]),
                    "gap_minutes": float(diff / pd.Timedelta(minutes=1)),
                }
            )
        return gaps

    def _sanity_checks(self, df: pd.DataFrame) -> dict[str, int]:
        invalid_price = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        invalid_volume = int((df["volume"] < 0).sum())
        return {"invalid_price_rows": invalid_price, "invalid_volume_rows": invalid_volume}

    def _outliers(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        pct = df["close"].pct_change().abs()
        return int((pct > 0.5).sum())

    def _expected_records(self, df: pd.DataFrame, interval: str) -> int:
        if df.empty:
            return 0
        df = normalize_open_time_ms_dataframe(df)
        minutes = get_interval_minutes(interval)
        start = pd.to_datetime(df["open_time"].min(), unit="ms", utc=True)
        end = pd.to_datetime(df["open_time"].max(), unit="ms", utc=True)
        return int(((end - start) / pd.Timedelta(minutes=minutes)) + 1)

    def _quality_score(
        self, gaps: list[dict[str, Any]], sanity: dict[str, int], outliers: int, missing_count: int
    ) -> int:
        score = 100
        score -= min(40, len(gaps) * 3)
        score -= min(20, sanity["invalid_price_rows"] * 2 + sanity["invalid_volume_rows"] * 2)
        score -= min(20, outliers)
        score -= min(20, missing_count)
        return max(0, score)

    def _quality_label(self, score: int) -> str:
        if score >= 90:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 50:
            return "Fair"
        return "Poor"

    def validate_symbol(self, symbol: str, interval: str) -> dict[str, Any]:
        """Validate one symbol/interval parquet file."""
        path = self.merged_dir / symbol / f"{symbol}-{interval}.parquet"
        if not path.exists():
            return {"symbol": symbol, "interval": interval, "exists": False}
        df = pd.read_parquet(path)
        df = normalize_open_time_ms_dataframe(df)
        gaps = self.check_data_continuity(df, interval)
        sanity = self._sanity_checks(df)
        outlier_count = self._outliers(df)
        expected = self._expected_records(df, interval)
        missing_count = max(0, expected - len(df))
        score = self._quality_score(gaps, sanity, outlier_count, missing_count)
        return {
            "symbol": symbol,
            "interval": interval,
            "exists": True,
            "rows": int(len(df)),
            "expected_rows": int(expected),
            "missing_rows": int(missing_count),
            "gaps": gaps,
            "outliers": int(outlier_count),
            "sanity": sanity,
            "quality_score": int(score),
            "quality_label": self._quality_label(score),
        }

    def validate_all(self, symbols: list[str], intervals: list[str]) -> pd.DataFrame:
        """Validate all requested symbol/interval combinations."""
        records: list[dict[str, Any]] = []
        jobs = [(symbol, interval) for symbol in symbols for interval in intervals]
        for symbol, interval in progress_bar(jobs, desc="Validating datasets", total=len(jobs)):
            records.append(self.validate_symbol(symbol, interval))
        return pd.DataFrame(records)

    def suggest_fixes(self, validation_results: dict[str, Any]) -> list[str]:
        """Suggest fixes based on validation issues."""
        suggestions: list[str] = []
        if validation_results.get("missing_rows", 0) > 0:
            suggestions.append("Run incremental downloader to fetch missing dates.")
        if validation_results.get("outliers", 0) > 0:
            suggestions.append("Verify outlier candles against Binance live API before filtering.")
        sanity = validation_results.get("sanity", {})
        if sanity.get("invalid_price_rows", 0) > 0:
            suggestions.append("Drop or correct rows with non-positive OHLC values.")
        if sanity.get("invalid_volume_rows", 0) > 0:
            suggestions.append("Remove rows with negative volume and re-merge source files.")
        if not suggestions:
            suggestions.append("No major issues detected.")
        return suggestions

    def _plot_missing_heatmap(self, df: pd.DataFrame) -> Path:
        data = df[["symbol", "interval", "missing_rows"]].copy()
        if data.empty:
            data = pd.DataFrame([{"symbol": "N/A", "interval": "N/A", "missing_rows": 0}])
        pivot = data.pivot_table(
            index="symbol", columns="interval", values="missing_rows", aggfunc="sum", fill_value=0
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds")
        plt.title("Missing Rows Heatmap")
        out = self.report_dir / "missing_heatmap.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def _plot_record_distribution(self, df: pd.DataFrame) -> Path:
        plt.figure(figsize=(12, 6))
        plot_df = df[df.get("exists", True) == True].copy()  # noqa: E712
        if "symbol" in plot_df.columns:
            plot_df["symbol_display"] = plot_df["symbol"].map(self._safe_plot_symbol)
            x_col = "symbol_display"
        else:
            x_col = "symbol"
        sns.barplot(data=plot_df, x=x_col, y="rows", hue="interval")
        plt.title("Record Count by Symbol")
        plt.xticks(rotation=45)
        out = self.report_dir / "record_distribution.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def _plot_quality_by_interval(self, df: pd.DataFrame) -> Path:
        plt.figure(figsize=(8, 5))
        agg = df.groupby("interval", as_index=False)["quality_score"].mean()
        sns.barplot(data=agg, x="interval", y="quality_score")
        plt.ylim(0, 100)
        plt.title("Average Quality by Interval")
        out = self.report_dir / "quality_by_interval.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def export_reports(self, results_df: pd.DataFrame, filename_prefix: str = "validation") -> dict[str, str]:
        """Export JSON/CSV/Markdown/HTML reports with charts."""
        ensure_dir(self.report_dir)
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\)")
        if "quality_label" not in results_df.columns:
            results_df = results_df.copy()
            results_df["quality_label"] = results_df.get("exists", False).map(
                lambda x: "Unavailable" if not x else "Unknown"
            )
        if "quality_score" not in results_df.columns:
            results_df = results_df.copy()
            results_df["quality_score"] = 0
        if "missing_rows" not in results_df.columns:
            results_df = results_df.copy()
            results_df["missing_rows"] = 0
        if "rows" not in results_df.columns:
            results_df = results_df.copy()
            results_df["rows"] = 0
        json_path = self.report_dir / f"{filename_prefix}.json"
        csv_path = self.report_dir / f"{filename_prefix}.csv"
        md_path = self.report_dir / f"{filename_prefix}.md"
        html_path = self.report_dir / f"{filename_prefix}.html"

        lines = ["# Validation Summary", "", f"Total records: {len(results_df)}", ""]
        for _, row in results_df.iterrows():
            lines.append(
                f"- {row['symbol']} {row['interval']}: {row['quality_label']} ({row['quality_score']})"
            )

        report_steps = [
            "json",
            "csv",
            "markdown",
            "heatmap",
            "record_distribution",
            "quality_plot",
            "html",
        ]
        heatmap: Path | None = None
        record_dist: Path | None = None
        quality_plot: Path | None = None
        for step in progress_bar(report_steps, desc="Generating reports", total=len(report_steps)):
            if step == "json":
                json_path.write_text(results_df.to_json(orient="records", indent=2), encoding="utf-8")
            elif step == "csv":
                results_df.to_csv(csv_path, index=False, encoding="utf-8")
            elif step == "markdown":
                md_path.write_text("\n".join(lines), encoding="utf-8")
            elif step == "heatmap":
                heatmap = self._plot_missing_heatmap(results_df)
            elif step == "record_distribution":
                record_dist = self._plot_record_distribution(results_df)
            elif step == "quality_plot":
                quality_plot = self._plot_quality_by_interval(results_df)
            elif step == "html":
                html_content = f"""
<html>
  <head><title>Validation Dashboard</title></head>
  <body>
    <h1>Validation Dashboard</h1>
    <h2>Overview</h2>
    {results_df.to_html(index=False)}
    <h2>Missing Data Heatmap</h2>
    <img src="{heatmap.name if heatmap else 'missing_heatmap.png'}" alt="Missing heatmap"/>
    <h2>Record Distribution</h2>
    <img src="{record_dist.name if record_dist else 'record_distribution.png'}" alt="Record distribution"/>
    <h2>Quality by Interval</h2>
    <img src="{quality_plot.name if quality_plot else 'quality_by_interval.png'}" alt="Quality by interval"/>
  </body>
</html>
"""
                html_path.write_text(html_content, encoding="utf-8")
        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "html": str(html_path),
        }
