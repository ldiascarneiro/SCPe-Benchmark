#!/usr/bin/env python3
"""Gera gráficos de benchmark entre modelos a partir de métricas do projeto.

Fontes suportadas:
- dados/output/metrics_history.jsonl (padrão)
- arquivos 05_generation_metrics.json dentro de dados/output/generation_*/
- arquivo adicional opcional (.csv/.json/.jsonl) com métricas extras

Saídas:
- PNGs dos gráficos na pasta definida em --output-dir
- benchmark_aggregated_metrics.csv com as médias por modelo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = PROJECT_ROOT / "dados" / "output" / "metrics_history.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "dados" / "input" / "SCPe_Dataset.csv"
DEFAULT_OUTPUT_SCAN_DIR = PROJECT_ROOT / "dados" / "output"


FIELD_ALIASES = {
    "run_id": ("run_id", "id", "session_id"),
    "session_dir": ("session_dir", "run_dir"),
    "model": ("model", "benchmark_model_id", "llm_model_name", "model_name"),
    "dataset_row_index": ("dataset_row_index", "row_index"),
    "generation_seconds": ("generation_seconds", "avg_time_s", "time_s", "time"),
    "total_tokens": (
        "generation_total_tokens",
        "total_tokens",
        "avg_total_tokens",
        "token_usage_total",
    ),
    "textual_quality_score": (
        "textual_quality_score",
        "avg_geval_quality",
        "geval_quality",
        "quality",
        "textual_quality",
    ),
    "dataset_adherence_score": (
        "dataset_adherence_score",
        "avg_geval_adherence",
        "geval_adherence",
        "adherence",
        "dataset_adherence",
    ),
    "bertscore_f1": ("bertscore_f1", "bertscore", "bert_f1", "avg_bertscore_f1"),
    "length_ratio": ("length_ratio", "len_ratio", "avg_length_ratio"),
    "tflops_required": (
        "estimated_theoretical_tflops_required",
        "estimated_tflops_required",
        "tflops_required",
        "avg_tflops_required",
    ),
    "tflops_per_second": (
        "estimated_theoretical_tflops_per_second",
        "estimated_tflops_per_second",
        "tflops_per_second",
        "avg_tflops_per_second",
    ),
    "had_retry_flag": ("had_retry", "generation_had_retry"),
    "had_completion_saturation_flag": ("had_completion_saturation", "generation_had_completion_saturation"),
    "had_truncation_suspected_flag": ("had_truncation_suspected", "generation_had_truncation_suspected"),
    "output_file": ("output_file", "generated_file", "generated_text_path"),
}


NUMERIC_FIELDS = (
    "generation_seconds",
    "total_tokens",
    "textual_quality_score",
    "dataset_adherence_score",
    "bertscore_f1",
    "length_ratio",
    "tflops_required",
    "tflops_per_second",
    "had_retry_flag",
    "had_completion_saturation_flag",
    "had_truncation_suspected_flag",
)


MARKERS = ["o", "X", "s", "P", "D", "d", "^", "v", "*", "h", "<", ">", "8", "p"]


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _info(message: str) -> None:
    print(f"[INFO] {message}")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, str):
        clean = value.strip().replace(",", ".")
        if not clean:
            return None
        try:
            parsed = float(clean)
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _bool_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) != 0)
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"1", "true", "yes", "y", "sim"}:
            return 1
        if clean in {"0", "false", "no", "n", "nao", "não"}:
            return 0
    return None


def _normalize_model_name(model: str) -> str:
    return model.replace(":", "_").strip()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_first(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _extract_from_metric_record(record: Dict[str, Any], source_path: Optional[Path] = None) -> Dict[str, Any]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    token_usage = record.get("generation_token_usage") or metrics.get("generation_token_usage") or {}
    generation_conditions = record.get("generation_conditions")
    if not isinstance(generation_conditions, dict):
        generation_conditions = metrics.get("generation_conditions")
        if not isinstance(generation_conditions, dict):
            generation_conditions = (metrics.get("generation_perf") or {}).get("generation_conditions") or {}

    run_id = record.get("session_dir") or record.get("metrics_file")
    if not run_id and source_path:
        run_id = str(source_path.parent.resolve())
    elif not run_id and source_path is None:
        run_id = f"line_{id(record)}"

    textual_quality = record.get("textual_quality_score")
    if textual_quality is None and isinstance(metrics.get("textual_quality"), dict):
        textual_quality = metrics["textual_quality"].get("score")

    dataset_adherence = record.get("dataset_adherence_score")
    if dataset_adherence is None and isinstance(metrics.get("dataset_adherence"), dict):
        dataset_adherence = metrics["dataset_adherence"].get("score")

    total_tokens = record.get("generation_total_tokens")
    if total_tokens is None:
        total_tokens = token_usage.get("total_tokens")

    session_dir = record.get("session_dir")
    output_file = record.get("output_file")

    if output_file is None and session_dir:
        session_path = Path(session_dir)
        candidates = sorted(session_path.glob("doc_generated_*.txt"))
        if candidates:
            output_file = str(candidates[0])
    elif output_file is None and source_path:
        candidates = sorted(source_path.parent.glob("doc_generated_*.txt"))
        if candidates:
            output_file = str(candidates[0])

    model_raw = record.get("benchmark_model_id") or record.get("llm_model_name")
    dataset_row_index = record.get("dataset_row_index")
    if dataset_row_index is None and isinstance(metrics, dict):
        dataset_row_index = metrics.get("dataset_row_index")

    return {
        "run_id": str(run_id),
        "session_dir": session_dir,
        "model": _normalize_model_name(str(model_raw)) if model_raw else None,
        "dataset_row_index": _safe_int(dataset_row_index),
        "generation_seconds": _safe_float(record.get("generation_seconds") or metrics.get("generation_seconds")),
        "total_tokens": _safe_float(total_tokens),
        "textual_quality_score": _safe_float(textual_quality),
        "dataset_adherence_score": _safe_float(dataset_adherence),
        "bertscore_f1": _safe_float(_extract_first(record, FIELD_ALIASES["bertscore_f1"])),
        "length_ratio": _safe_float(_extract_first(record, FIELD_ALIASES["length_ratio"])),
        "tflops_required": _safe_float(
            _extract_first(record, FIELD_ALIASES["tflops_required"])
            or _extract_first(metrics, FIELD_ALIASES["tflops_required"])
        ),
        "tflops_per_second": _safe_float(
            _extract_first(record, FIELD_ALIASES["tflops_per_second"])
            or _extract_first(metrics, FIELD_ALIASES["tflops_per_second"])
        ),
        "had_retry_flag": _bool_to_int(
            _extract_first(record, FIELD_ALIASES["had_retry_flag"]) or generation_conditions.get("had_retry")
        ),
        "had_completion_saturation_flag": _bool_to_int(
            _extract_first(record, FIELD_ALIASES["had_completion_saturation_flag"])
            or generation_conditions.get("had_completion_saturation")
        ),
        "had_truncation_suspected_flag": _bool_to_int(
            _extract_first(record, FIELD_ALIASES["had_truncation_suspected_flag"])
            or generation_conditions.get("had_truncation_suspected")
        ),
        "output_file": output_file,
    }


def _extract_generic_row(row: Dict[str, Any], source_name: str, index: int) -> Dict[str, Any]:
    model_value = _extract_first(row, FIELD_ALIASES["model"])
    run_id = _extract_first(row, FIELD_ALIASES["run_id"]) or f"{source_name}:{index}"

    return {
        "run_id": str(run_id),
        "session_dir": _extract_first(row, FIELD_ALIASES["session_dir"]),
        "model": _normalize_model_name(str(model_value)) if model_value else None,
        "dataset_row_index": _safe_int(_extract_first(row, FIELD_ALIASES["dataset_row_index"])),
        "generation_seconds": _safe_float(_extract_first(row, FIELD_ALIASES["generation_seconds"])),
        "total_tokens": _safe_float(_extract_first(row, FIELD_ALIASES["total_tokens"])),
        "textual_quality_score": _safe_float(_extract_first(row, FIELD_ALIASES["textual_quality_score"])),
        "dataset_adherence_score": _safe_float(_extract_first(row, FIELD_ALIASES["dataset_adherence_score"])),
        "bertscore_f1": _safe_float(_extract_first(row, FIELD_ALIASES["bertscore_f1"])),
        "length_ratio": _safe_float(_extract_first(row, FIELD_ALIASES["length_ratio"])),
        "tflops_required": _safe_float(_extract_first(row, FIELD_ALIASES["tflops_required"])),
        "tflops_per_second": _safe_float(_extract_first(row, FIELD_ALIASES["tflops_per_second"])),
        "had_retry_flag": _bool_to_int(_extract_first(row, FIELD_ALIASES["had_retry_flag"])),
        "had_completion_saturation_flag": _bool_to_int(
            _extract_first(row, FIELD_ALIASES["had_completion_saturation_flag"])
        ),
        "had_truncation_suspected_flag": _bool_to_int(
            _extract_first(row, FIELD_ALIASES["had_truncation_suspected_flag"])
        ),
        "output_file": _extract_first(row, FIELD_ALIASES["output_file"]),
    }


def _merge_run(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _load_runs_from_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        _warn(f"Arquivo de histórico não encontrado: {path}")
        return []

    runs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                record = json.loads(clean)
            except json.JSONDecodeError:
                _warn(f"Linha inválida ignorada em {path}:{i}")
                continue
            runs.append(_extract_from_metric_record(record))
    _info(f"Carregadas {len(runs)} execuções de {path}")
    return runs


def _load_runs_from_metrics_scan(output_dir: Path) -> List[Dict[str, Any]]:
    if not output_dir.exists():
        return []

    metric_files = sorted(output_dir.glob("generation_*/05_generation_metrics.json"))
    runs: List[Dict[str, Any]] = []
    for metric_file in metric_files:
        try:
            data = _read_json(metric_file)
        except Exception as exc:  # pylint: disable=broad-except
            _warn(f"Falha ao ler {metric_file}: {exc}")
            continue
        runs.append(_extract_from_metric_record(data, source_path=metric_file))
    _info(f"Carregadas {len(runs)} execuções por varredura de {output_dir}")
    return runs


def _detect_csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;#\t")
    except csv.Error:
        class _Fallback(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            escapechar = None
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Fallback()


def _load_rows_from_extra_input(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo --input não encontrado: {path}")

    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(_extract_generic_row(data, str(path), i))
        _info(f"Carregadas {len(rows)} linhas extras de {path}")
        return rows

    if suffix == ".json":
        data = _read_json(path)
        if isinstance(data, dict):
            if isinstance(data.get("rows"), list):
                source_rows = data["rows"]
            elif isinstance(data.get("data"), list):
                source_rows = data["data"]
            else:
                source_rows = [data]
        elif isinstance(data, list):
            source_rows = data
        else:
            source_rows = []

        for i, row in enumerate(source_rows):
            if isinstance(row, dict):
                rows.append(_extract_generic_row(row, str(path), i))
        _info(f"Carregadas {len(rows)} linhas extras de {path}")
        return rows

    dialect = _detect_csv_dialect(path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader):
            rows.append(_extract_generic_row(row, str(path), i))
    _info(f"Carregadas {len(rows)} linhas extras de {path}")
    return rows


def _read_text_word_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return len(text.split())


def _load_dataset_word_counts(dataset_path: Path) -> Dict[int, int]:
    if not dataset_path.exists():
        _warn(f"Dataset não encontrado para calcular length_ratio: {dataset_path}")
        return {}

    row_counts: Dict[int, int] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="#")
        for idx, row in enumerate(reader):
            text = row.get("TEXTO_ORIGINAL", "") or ""
            row_counts[idx] = len(text.split())
    return row_counts


def _fill_missing_length_ratio(runs_by_id: Dict[str, Dict[str, Any]], dataset_path: Path) -> None:
    missing = [r for r in runs_by_id.values() if r.get("length_ratio") is None]
    if not missing:
        return

    dataset_word_counts = _load_dataset_word_counts(dataset_path)
    if not dataset_word_counts:
        return

    filled = 0
    for run in missing:
        row_idx = run.get("dataset_row_index")
        if row_idx is None:
            continue
        original_wc = dataset_word_counts.get(row_idx)
        if not original_wc:
            continue

        output_file = run.get("output_file")
        if output_file:
            generated_wc = _read_text_word_count(Path(output_file))
        else:
            session_dir = run.get("session_dir")
            generated_wc = None
            if session_dir:
                candidates = sorted(Path(session_dir).glob("doc_generated_*.txt"))
                if candidates:
                    generated_wc = _read_text_word_count(candidates[0])

        if generated_wc is None:
            continue

        run["length_ratio"] = generated_wc / original_wc
        filled += 1

    if filled:
        _info(f"Length ratio calculado automaticamente para {filled} execução(ões)")


def _aggregate_by_model(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        model = run.get("model")
        if not model:
            continue

        metric_lists = {field: [] for field in NUMERIC_FIELDS}
        bucket = grouped.setdefault(
            model,
            {
                "model": model,
                "runs": 0,
                **metric_lists,
            },
        )

        bucket["runs"] += 1
        for field in NUMERIC_FIELDS:
            value = run.get(field)
            if value is not None:
                bucket[field].append(value)

    aggregated: List[Dict[str, Any]] = []
    for model, data in grouped.items():
        row: Dict[str, Any] = {"model": model, "runs": data["runs"]}
        for field in NUMERIC_FIELDS:
            values = data[field]
            row[field] = (sum(values) / len(values)) if values else None
        aggregated.append(row)

    return aggregated


def _set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:  # pylint: disable=broad-except
        plt.style.use("ggplot")
    plt.rcParams["figure.facecolor"] = "#f3f3f3"
    plt.rcParams["axes.facecolor"] = "#ededed"
    plt.rcParams["axes.edgecolor"] = "#bbbbbb"
    plt.rcParams["grid.alpha"] = 0.45
    plt.rcParams["grid.color"] = "#aaaaaa"


def _viridis_colors(size: int) -> List[Any]:
    if size <= 1:
        return [plt.cm.viridis(0.5)]
    return [plt.cm.viridis(0.05 + 0.9 * (i / (size - 1))) for i in range(size)]


def _plot_bar(
    rows: List[Dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    ascending: bool = False,
    decimals: int = 2,
) -> bool:
    filtered = [r for r in rows if r.get(metric) is not None]
    if not filtered:
        _warn(f"Sem dados para gráfico: {title}")
        return False

    filtered.sort(key=lambda r: r[metric], reverse=not ascending)
    labels = [r["model"] for r in filtered]
    values = [r[metric] for r in filtered]
    colors = _viridis_colors(len(values))

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, value in zip(bars, values):
        y = bar.get_height()
        va = "bottom"
        offset = max(abs(value) * 0.005, 0.01)
        if y < 0:
            va = "top"
            offset = -offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=40)
    if min(values) < 0 < max(values):
        ax.axhline(0, color="gray", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    _info(f"Gráfico salvo: {output_path}")
    return True


def _plot_scatter(
    rows: List[Dict[str, Any]],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    filtered = [r for r in rows if r.get(x_metric) is not None and r.get(y_metric) is not None]
    if not filtered:
        _warn(f"Sem dados para gráfico: {title}")
        return False

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, row in enumerate(sorted(filtered, key=lambda r: r["model"])):
        marker = MARKERS[idx % len(MARKERS)]
        color = colors[idx % len(colors)]
        ax.scatter(
            row[x_metric],
            row[y_metric],
            s=85,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidth=0.7,
            alpha=0.95,
            label=row["model"],
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _info(f"Gráfico salvo: {output_path}")
    return True


def _save_aggregated_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "runs",
        "generation_seconds",
        "total_tokens",
        "textual_quality_score",
        "dataset_adherence_score",
        "bertscore_f1",
        "length_ratio",
        "tflops_required",
        "tflops_per_second",
        "had_retry_flag",
        "had_completion_saturation_flag",
        "had_truncation_suspected_flag",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _info(f"Métricas agregadas salvas: {path}")


def generate_graphs(
    history_path: Path,
    output_dir: Path,
    dataset_path: Path,
    extra_input: Optional[Path],
    scan_metrics_files: bool,
    scan_dir: Path,
) -> int:
    _set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_by_id: Dict[str, Dict[str, Any]] = {}

    for run in _load_runs_from_history(history_path):
        run_id = run["run_id"]
        runs_by_id[run_id] = _merge_run(runs_by_id.get(run_id, {}), run)

    if scan_metrics_files:
        for run in _load_runs_from_metrics_scan(scan_dir):
            run_id = run["run_id"]
            runs_by_id[run_id] = _merge_run(runs_by_id.get(run_id, {}), run)

    if extra_input:
        for run in _load_rows_from_extra_input(extra_input):
            run_id = run["run_id"]
            runs_by_id[run_id] = _merge_run(runs_by_id.get(run_id, {}), run)

    if not runs_by_id:
        _warn("Nenhuma execução encontrada para gerar gráficos.")
        return 1

    _fill_missing_length_ratio(runs_by_id, dataset_path)
    aggregated_rows = _aggregate_by_model(list(runs_by_id.values()))

    if not aggregated_rows:
        _warn("Não foi possível agregar dados por modelo.")
        return 1

    aggregated_rows.sort(key=lambda r: r["model"])
    _save_aggregated_csv(aggregated_rows, output_dir / "benchmark_aggregated_metrics.csv")

    generated_count = 0
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="total_tokens",
            title="Average Total Tokens by Model",
            ylabel="Total Tokens",
            output_path=output_dir / "avg_total_tokens_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="generation_seconds",
            title="Average Time (s) by Model",
            ylabel="Time (s)",
            output_path=output_dir / "avg_time_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="textual_quality_score",
            title="Average G-Eval (Quality) by Model",
            ylabel="G-Eval (Quality)",
            output_path=output_dir / "avg_geval_quality_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="dataset_adherence_score",
            title="Average G-Eval (Adherence) by Model",
            ylabel="G-Eval (Adherence)",
            output_path=output_dir / "avg_geval_adherence_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="bertscore_f1",
            title="Average BERTScore (F1) by Model",
            ylabel="BERTScore (F1)",
            output_path=output_dir / "avg_bertscore_f1_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="length_ratio",
            title="Average Length Ratio by Model",
            ylabel="Length Ratio",
            output_path=output_dir / "avg_length_ratio_by_model.png",
            ascending=False,
            decimals=2,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="tflops_per_second",
            title="Average Theoretical TFLOPs/s by Model",
            ylabel="Theoretical TFLOPs/s",
            output_path=output_dir / "avg_tflops_per_second_by_model.png",
            ascending=False,
            decimals=3,
        )
    )
    generated_count += int(
        _plot_bar(
            aggregated_rows,
            metric="tflops_required",
            title="Average Total Theoretical TFLOPs Required by Model",
            ylabel="Estimated Theoretical TFLOPs Required",
            output_path=output_dir / "avg_tflops_required_by_model.png",
            ascending=False,
            decimals=3,
        )
    )
    generated_count += int(
        _plot_scatter(
            aggregated_rows,
            x_metric="generation_seconds",
            y_metric="textual_quality_score",
            title="Efficiency: Quality vs Time",
            xlabel="Average Time (s)",
            ylabel="Average G-Eval (Quality)",
            output_path=output_dir / "efficiency_quality_vs_time.png",
        )
    )
    generated_count += int(
        _plot_scatter(
            aggregated_rows,
            x_metric="total_tokens",
            y_metric="textual_quality_score",
            title="Efficiency: Quality vs Token Usage",
            xlabel="Average Total Tokens",
            ylabel="Average G-Eval (Quality)",
            output_path=output_dir / "efficiency_quality_vs_token_usage.png",
        )
    )

    _info(f"Total de gráficos gerados: {generated_count}")
    return 0 if generated_count > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera gráficos de benchmark entre modelos.")
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help=f"Arquivo de histórico JSONL (padrão: {DEFAULT_HISTORY})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Arquivo adicional (.csv/.json/.jsonl) com métricas por execução/modelo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída dos gráficos (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset para cálculo de length_ratio (padrão: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=DEFAULT_OUTPUT_SCAN_DIR,
        help=f"Pasta para varrer arquivos 05_generation_metrics.json (padrão: {DEFAULT_OUTPUT_SCAN_DIR})",
    )
    parser.add_argument(
        "--no-scan-metrics-files",
        action="store_true",
        help="Desativa varredura de arquivos generation_*/05_generation_metrics.json.",
    )
    args = parser.parse_args()

    return generate_graphs(
        history_path=args.history,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        extra_input=args.input,
        scan_metrics_files=not args.no_scan_metrics_files,
        scan_dir=args.scan_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
