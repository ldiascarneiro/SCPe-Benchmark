# Created by Leandro Carneiro at 18/09/2025
# Description: 
# -------------------------------------------------------------

# graph/nodes/planner_node.py
from typing import Dict, Any
import json
from services_packages.generation_utils import SessionManager, load_case_from_csv


def _build_csv_plan() -> Dict[str, Any]:
    # Plano fixo para modo CSV: usa apenas DESC_LOCAL, VEST_LOCAL e VEST_CADAVER.
    return {
        "style_profile": {
            "tone": "técnico, impessoal, jurídico-formal",
            "format_conventions": [
                "somente fatos derivados de DESC_LOCAL, VEST_LOCAL e VEST_CADAVER",
                "não inventar datas, horários, pessoas ou numerações não informadas",
                "quando faltar dado, declarar explicitamente 'não disponível nos insumos'",
            ],
            "terminology": [],
        },
        "guide_templates": {},
        "sections": [
            {"id": "1", "title": "HISTÓRICO", "children": []},
            {
                "id": "2",
                "title": "EXAMES",
                "children": [
                    {"title": "DO LOCAL", "children": []},
                    {"title": "VESTÍGIOS DO LOCAL", "children": []},
                    {"title": "VESTÍGIOS DO CADÁVER", "children": []},
                ],
            },
            {"id": "3", "title": "ANÁLISE", "children": []},
            {"id": "4", "title": "CONCLUSÃO", "children": []},
        ],
    }


def planner_node(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    state = state or {}
    requested_row_index = state.get("dataset_row_index")
    benchmark_model_id = state.get("benchmark_model_id")
    llm_model_name = state.get("llm_model_name")

    sess = SessionManager()
    suffix_parts: list[str] = []
    if requested_row_index is not None:
        suffix_parts.append(f"row_{requested_row_index}")
    if benchmark_model_id:
        suffix_parts.append(f"model_{benchmark_model_id}")
    suffix = "_".join(suffix_parts) if suffix_parts else None
    sess.start(suffix=suffix)
    sess.log({
        "step": "start",
        "benchmark_model_id": benchmark_model_id,
        "llm_model_name": llm_model_name,
    })

    case = load_case_from_csv(sess, row_index=requested_row_index)
    sess.log({"step": "csv_case_loaded", "row_index": case["case_metadata"]["row_index"]})

    plan = _build_csv_plan()
    plan_trace = {
        "source": "SCPe_Dataset.csv",
        "columns_used": ["DESC_LOCAL", "VEST_LOCAL", "VEST_CADAVER"],
        "row_index": case["case_metadata"]["row_index"],
        "benchmark_model_id": benchmark_model_id,
        "llm_model_name": llm_model_name,
    }
    planner_warnings: list[str] = []

    sess.save_text("01_plan.json", json.dumps(plan, ensure_ascii=False, indent=2))
    sess.save_json("02_plan_trace.json", plan_trace)
    sess.log({"step": "planner_done"})

    return {
        "plan": plan,
        "plan_trace": plan_trace,
        "planner_warnings": planner_warnings,
        "input_perito": case["input_perito"],
        "docs_complementares": [],
        "guia_conteudo": "",
        "session_dir": str(sess.session_dir),
        "writer_attempt": 0,
        "writer_feedback": [],
        "dataset_row_index": case["case_metadata"]["row_index"],
        "benchmark_model_id": benchmark_model_id,
        "llm_model_name": llm_model_name,
    }
