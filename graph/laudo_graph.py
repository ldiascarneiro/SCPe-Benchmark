# graph/laudo_graph.py
import csv
from pathlib import Path
import traceback
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import config
import datetime

# Importa os nós refatorados
from graph.nodes.planner_node import planner_node
from graph.nodes.writer_node import writer_node
from graph.nodes.evoluator_node import evoluator_node
from graph.nodes.save_node import save_node
from services_packages.llm import ensure_ollama_models_available

class LaudoState(TypedDict, total=False):
    plan: Dict[str, Any]
    input_perito: str
    docs_complementares: List[str]
    sections_texts: List[Dict[str, str]]
    laudo_gerado: str
    status: str
    limit_top_sections: int
    preview_only: bool
    guia_conteudo: str
    session_dir: str
    plan_trace: Dict[str, Any]
    planner_warnings: List[str]
    writer_attempt: int
    writer_feedback: List[str]
    laudo_path: str
    dataset_row_index: int
    quality_report: Dict[str, Any]
    generation_perf: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    llm_model_name: str
    benchmark_model_id: str

builder = StateGraph(LaudoState)

builder.add_node("planner_node", planner_node)
builder.add_node("writer_node", writer_node)
builder.add_node("evoluator_node", evoluator_node)
builder.add_node("save", save_node)

builder.set_entry_point("planner_node")
builder.add_edge("planner_node", "writer_node")
builder.add_edge("writer_node", "evoluator_node")
builder.add_edge("evoluator_node", "save")
builder.add_edge("save", END)

laudo_graph = builder.compile()


def _resolve_benchmark_models() -> List[str]:
    raw_models = getattr(config, "BENCHMARK_MODELS", None)
    models: List[str] = []

    if isinstance(raw_models, list):
        for item in raw_models:
            model_name = str(item).strip()
            if model_name:
                models.append(model_name)

    if not models:
        model_name = str(getattr(config, "LLM_MODEL_NAME", "")).strip()
        models = [model_name]

    return models


def _count_dataset_rows() -> int:
    csv_path = Path(getattr(config, "DATASET_CSV_PATH", ""))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    delimiter = str(getattr(config, "CSV_DELIMITER", "#"))
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader, None)  # cabeçalho
        return sum(1 for _ in reader)


def run_graph(
    limit_top_sections: int = 3,
    preview_only: bool = True,
    start_row: int = 0,
    num_rows: int = 1,
):
    if start_row < 0:
        raise ValueError("start_row deve ser >= 0.")

    if num_rows == -1:
        total_dataset_rows = _count_dataset_rows()
        if start_row >= total_dataset_rows:
            raise ValueError(
                f"start_row={start_row} fora do dataset; total de linhas={total_dataset_rows}."
            )
        num_rows = total_dataset_rows - start_row
    elif num_rows < 1:
        raise ValueError("num_rows deve ser >= 1 ou -1 para processar até o fim do dataset.")

    benchmark_models = _resolve_benchmark_models()
    provider = str(getattr(config, "LLM_PROVIDER", "")).strip().lower()
    if provider == "ollama":
        print("Validando disponibilidade dos modelos no Ollama...")
        ensure_ollama_models_available(benchmark_models)

    total_runs = num_rows * len(benchmark_models)
    continue_on_error = bool(getattr(config, "BENCHMARK_CONTINUE_ON_ERROR", True))
    model_labels = ", ".join(benchmark_models)
    print(
        f"Executando benchmark para {num_rows} linha(s), a partir da linha {start_row}, "
        f"em {len(benchmark_models)} modelo(s): {model_labels}"
    )
    run_idx = 0
    failed_runs = 0
    for model_name in benchmark_models:
        for offset in range(num_rows):
            row_index = start_row + offset
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] Processando linha {row_index} do dataset "
                f"com modelo {model_name} [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]..."
            )
            try:
                laudo_graph.invoke({
                    "limit_top_sections": limit_top_sections,
                    "preview_only": preview_only,
                    "dataset_row_index": row_index,
                    "llm_model_name": model_name,
                    "benchmark_model_id": model_name,
                })
            except Exception as exc:
                failed_runs += 1
                print(f"[ERROR] Falha no run linha={row_index}, modelo={model_name}: {exc}")
                print(traceback.format_exc(limit=1).strip())
                if not continue_on_error:
                    raise
                print("[WARN] Continuando benchmark (BENCHMARK_CONTINUE_ON_ERROR=True).")

    if failed_runs:
        print(f"Benchmark finalizado com falhas: {failed_runs}/{total_runs} execução(ões) falharam.")
    else:
        print(f"Benchmark finalizado com sucesso: {total_runs}/{total_runs} execução(ões).")
