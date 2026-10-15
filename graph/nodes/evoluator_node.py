from typing import Dict, Any, List
import json
import re

import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def _get_openai_api_key() -> str | None:
    key = getattr(config, "OPENAI_API_KEY", None)
    if key is None:
        return None
    key_str = str(key).strip()
    return key_str or None


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _extract_json_payload(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else {}
    except Exception:
        pass

    matches = re.findall(r"\{[\s\S]*\}", text)
    for m in matches:
        try:
            val = json.loads(m)
            if isinstance(val, dict):
                return val
        except Exception:
            continue
    return {}


def _clamp_score(score: Any) -> float | None:
    try:
        val = float(score)
    except Exception:
        return None
    return max(0.0, min(10.0, round(val, 3)))


def _run_geval(metric_name: str, prompt: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "metric": metric_name,
        "method": "g-eval",
        "model": getattr(config, "GEVAL_MODEL_NAME", "gpt-4o"),
        "score": None,
        "justification": "",
        "raw_response": "",
        "error": "",
    }
    try:
        llm = ChatOpenAI(
            model=getattr(config, "GEVAL_MODEL_NAME", "gpt-4o"),
            temperature=float(getattr(config, "GEVAL_TEMPERATURE", 0.0)),
            max_tokens=int(getattr(config, "GEVAL_MAX_OUTPUT_TOKENS", 600)),
            api_key=_get_openai_api_key(),
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = _to_text(getattr(response, "content", ""))
        payload = _extract_json_payload(raw)

        result["raw_response"] = raw
        result["score"] = _clamp_score(payload.get("score"))
        result["justification"] = str(payload.get("justification") or "").strip()

        if result["score"] is None:
            result["error"] = "Resposta sem score numérico válido."
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _geval_textual_quality(text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Você é avaliador especialista em qualidade textual de laudos periciais.
Avalie APENAS a qualidade textual do laudo abaixo e retorne uma nota de 0 a 10.

Critérios:
- Clareza, coerência e coesão.
- Organização e fluidez entre seções.
- Tom técnico, impessoal e jurídico-formal.
- Correção gramatical e precisão.

Plano esperado:
{json.dumps(plan.get("sections", []), ensure_ascii=False, indent=2)}

Laudo:
{text}

Responda SOMENTE em JSON:
{{
  "score": <numero de 0 a 10>,
  "justification": "<resumo curto>"
}}
""".strip()
    return _run_geval("textual_quality", prompt)


def _geval_dataset_adherence(text: str, dataset_input: str) -> Dict[str, Any]:
    prompt = f"""
Você é avaliador especialista em aderência factual a insumos.
Compare o laudo aos dados do dataset e atribua nota de 0 a 10.

Critérios:
- Cobertura das informações relevantes do dataset.
- Fidelidade aos fatos (sem distorção).
- Penalização de alucinações.
- Uso específico de detalhes do dataset.

Dados do dataset:
{dataset_input}

Laudo:
{text}

Responda SOMENTE em JSON:
{{
  "score": <numero de 0 a 10>,
  "justification": "<resumo curto>"
}}
""".strip()
    return _run_geval("dataset_adherence", prompt)


def _is_sumario_title(title: str) -> bool:
    t = str(title or "").strip().upper()
    return t in {"SUMÁRIO", "SUMARIO"}


def _section_content_word_count(section_text: str, sec_id: str, sec_title: str) -> int:
    text = (section_text or "").strip()
    if not text:
        return 0

    main_header = f"### {sec_id} – {str(sec_title or '').upper()}".strip()
    if text.startswith(main_header):
        text = text[len(main_header):].strip()

    text = re.sub(r"(?m)^\s*###\s+\d+(?:\.\d+)*\s+–\s*[^\n]*\n?", "", text).strip()
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return len(words)


def _analyze_structural_omissions(state: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    sections_generated = state.get("sections_texts") or []
    if not isinstance(sections_generated, list):
        sections_generated = []

    expected_top_sections = []
    for sec in plan.get("sections", []) or []:
        if not isinstance(sec, dict):
            continue
        sec_id = str(sec.get("id") or "").strip()
        sec_title = str(sec.get("title") or "").strip()
        if not sec_id or _is_sumario_title(sec_title):
            continue
        expected_top_sections.append({"id": sec_id, "title": sec_title})

    generated_map: Dict[str, Dict[str, Any]] = {}
    for sec in sections_generated:
        if not isinstance(sec, dict):
            continue
        sec_id = str(sec.get("id") or "").strip()
        if not sec_id or "." in sec_id:
            continue
        generated_map[sec_id] = sec

    min_words = int(getattr(config, "SECTION_MIN_CONTENT_WORDS", 80))
    empty_or_very_short_ids: List[str] = []
    missing_ids: List[str] = []

    for exp in expected_top_sections:
        sid = exp["id"]
        sec = generated_map.get(sid)
        if sec is None:
            missing_ids.append(sid)
            continue
        wc = _section_content_word_count(
            sec.get("text") or "",
            sec_id=sid,
            sec_title=sec.get("title") or exp["title"],
        )
        if wc < min_words:
            empty_or_very_short_ids.append(sid)

    penalty_per_section = float(getattr(config, "ADHERENCE_EMPTY_SECTION_PENALTY_PER_SECTION", 1.0))
    max_penalty = float(getattr(config, "ADHERENCE_OMISSION_MAX_PENALTY", 4.0))
    penalty = min(max_penalty, penalty_per_section * float(len(empty_or_very_short_ids) + len(missing_ids)))

    return {
        "expected_top_sections": len(expected_top_sections),
        "empty_or_very_short_sections": empty_or_very_short_ids,
        "missing_sections": missing_ids,
        "min_words_threshold": min_words,
        "penalty_applied": round(penalty, 3),
    }


def evoluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = (state.get("laudo_gerado") or "").strip()
    if not text:
        raise RuntimeError("evaluator: laudo_gerado ausente para avaliação de métricas.")

    plan = state.get("plan", {}) or {}
    generation_metrics = dict(state.get("generation_metrics", {}) or {})

    textual_quality = _geval_textual_quality(text, plan)
    dataset_adherence = _geval_dataset_adherence(text, state.get("input_perito", "") or "")

    omission_diag = _analyze_structural_omissions(state, plan)
    raw_adherence_score = dataset_adherence.get("score")
    if raw_adherence_score is not None:
        penalty = float(omission_diag.get("penalty_applied", 0.0))
        adjusted = max(0.0, round(float(raw_adherence_score) - penalty, 3))
        dataset_adherence["raw_score"] = raw_adherence_score
        dataset_adherence["score"] = adjusted
        dataset_adherence["structural_omission_penalty"] = penalty
        dataset_adherence["structural_omission_details"] = omission_diag
        if penalty > 0:
            note = (
                f" Penalização estrutural aplicada ({penalty:.3f}) por seções vazias/omitidas: "
                f"{omission_diag.get('empty_or_very_short_sections', []) + omission_diag.get('missing_sections', [])}."
            )
            dataset_adherence["justification"] = (dataset_adherence.get("justification") or "").strip() + note

    metric_errors: List[str] = []
    if textual_quality.get("score") is None:
        metric_errors.append(f"textual_quality inválido: {textual_quality.get('error') or 'sem score'}")
    if dataset_adherence.get("score") is None:
        metric_errors.append(f"dataset_adherence inválido: {dataset_adherence.get('error') or 'sem score'}")
    if metric_errors:
        raise RuntimeError(" | ".join(metric_errors))

    generation_metrics["textual_quality"] = textual_quality
    generation_metrics["dataset_adherence"] = dataset_adherence

    quality_report = {
        "status": "generated",
        "evaluation_enabled": True,
        "textual_quality_score": textual_quality.get("score"),
        "dataset_adherence_score": dataset_adherence.get("score"),
        "dataset_adherence_raw_score": dataset_adherence.get("raw_score"),
        "dataset_adherence_structural_penalty": dataset_adherence.get("structural_omission_penalty"),
        "structural_omission_details": omission_diag,
        "evaluation_errors": [x for x in [textual_quality.get("error"), dataset_adherence.get("error")] if x],
    }

    return {
        "quality_report": quality_report,
        "generation_metrics": generation_metrics,
        "status": "generated",
    }
