# Created by Leandro Carneiro at 18/09/2025
# Description: 
# -------------------------------------------------------------

# graph/nodes/save_node.py
import os
import re
import json
from typing import Dict, Any
from pathlib import Path
import config
from services_packages import general_functions as gf

def save_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw = state.get("session_dir")
    if not raw:
        raise RuntimeError("save_output: session_dir ausente.")
    session_dir = Path(str(raw)).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    text = state.get("laudo_gerado")
    if not text or not text.strip():
        raise RuntimeError("save_output: laudo está vazio.")

    session_name = os.path.basename(str(session_dir).rstrip("/"))
    m_ts = re.match(r"^generation_(\d{8}_\d{6})(?:_.+)?$", session_name)
    session_ts = m_ts.group(1) if m_ts else None

    filename = f"doc_generated_{session_ts}.txt"
    out_path = session_dir / filename
    gf.write_text(out_path, text)

    quality_report = state.get("quality_report", {})
    plan_trace = state.get("plan_trace", {})
    generation_metrics = state.get("generation_metrics", {})
    llm_model_name = str(
        state.get("llm_model_name")
        or generation_metrics.get("llm_model_name")
        or ""
    ).strip()
    benchmark_model_id = str(
        state.get("benchmark_model_id")
        or generation_metrics.get("benchmark_model_id")
        or llm_model_name
    ).strip()
    manifest = {
        "status": "generated",
        "approved": True,
        "writer_attempt": state.get("writer_attempt", 0),
        "output_file": str(out_path),
        "llm_model_name": llm_model_name,
        "benchmark_model_id": benchmark_model_id,
    }
    gf.write_json(session_dir / "quality_report.json", quality_report or {"status": "metrics_not_computed"})
    gf.write_json(session_dir / "plan_trace.json", plan_trace or {})
    metrics_path = session_dir / "05_generation_metrics.json"
    gf.write_json(metrics_path, generation_metrics or {})
    gf.write_json(session_dir / "generation_manifest.json", manifest)

    history_path = config.OUTPUT_DIR / "metrics_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_row = {
        "session_dir": str(session_dir),
        "status": manifest["status"],
        "llm_model_name": llm_model_name,
        "benchmark_model_id": benchmark_model_id,
        "dataset_row_index": generation_metrics.get("dataset_row_index"),
        "textual_quality_score": (generation_metrics.get("textual_quality") or {}).get("score"),
        "dataset_adherence_score": (generation_metrics.get("dataset_adherence") or {}).get("score"),
        "dataset_adherence_raw_score": (generation_metrics.get("dataset_adherence") or {}).get("raw_score"),
        "dataset_adherence_structural_penalty": (generation_metrics.get("dataset_adherence") or {}).get(
            "structural_omission_penalty"
        ),
        "generation_seconds": generation_metrics.get("generation_seconds"),
        "generation_total_tokens": (generation_metrics.get("generation_token_usage") or {}).get("total_tokens"),
        "estimated_tflops_required": generation_metrics.get("estimated_tflops_required"),
        "estimated_tflops_per_second": generation_metrics.get("estimated_tflops_per_second"),
        "estimated_theoretical_tflops_required": generation_metrics.get("estimated_theoretical_tflops_required"),
        "estimated_theoretical_tflops_per_second": generation_metrics.get("estimated_theoretical_tflops_per_second"),
        "had_retry": (generation_metrics.get("generation_conditions") or {}).get("had_retry"),
        "had_completion_saturation": (generation_metrics.get("generation_conditions") or {}).get(
            "had_completion_saturation"
        ),
        "had_truncation_suspected": (generation_metrics.get("generation_conditions") or {}).get(
            "had_truncation_suspected"
        ),
        "output_file": str(out_path),
        "metrics_file": str(metrics_path),
        "metrics": generation_metrics or {},
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_row, ensure_ascii=False) + "\n")

    print(f"New doc saved: {out_path}")

    return {
        "laudo_path": str(out_path),
        "laudo_gerado": text,
        "session_dir": str(session_dir),
        "status": manifest["status"],
    }
