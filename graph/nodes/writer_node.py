# Created by Leandro Carneiro at 18/09/2025
# Description:
# -------------------------------------------------------------
# Gera cada seção principal do laudo a partir do caso CSV-only
# (DESC_LOCAL, VEST_LOCAL, VEST_CADAVER), salvando output por seção.

from typing import Dict, Any, List
import json
import re
import hashlib
import time
from pathlib import Path

import config
from langchain_core.messages import HumanMessage

from services_packages.generation_utils import (
    normalize_title,
    heading,
    sanitize_section_output,
)
from services_packages.llm import load_llm, resolve_model_params_b_ollama
from services_packages import general_functions as gf


def _extract_case_fields(input_perito: str) -> Dict[str, str]:
    fields = {
        "DATA_DO_EXAME": "não disponível nos insumos",
        "PRIMEIRO_PERITO": "não disponível nos insumos",
        "SEGUNDO_PERITO": "não disponível nos insumos",
        "DELEGACIA_SOLICITANTE": "não disponível nos insumos",
        "ENDEREÇO": "não disponível nos insumos",
        "COORDENADAS_GEOGRAFICAS": "não disponível nos insumos",
    }
    patterns = {
        "DATA_DO_EXAME": r"(?im)^\s*Data do exame:\s*(.+)$",
        "PRIMEIRO_PERITO": r"(?im)^\s*Primeiro Perito:\s*(.+)$",
        "SEGUNDO_PERITO": r"(?im)^\s*Segundo Perito:\s*(.+)$",
        "DELEGACIA_SOLICITANTE": r"(?im)^\s*Delegacia Solicitante:\s*(.+)$",
        "ENDEREÇO": r"(?im)^\s*Endereço:\s*(.+)$",
        "COORDENADAS_GEOGRAFICAS": r"(?im)^\s*Coordenadas geográficas:\s*(.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, input_perito)
        if m:
            fields[key] = m.group(1).strip()
    return fields


def _render_template(template: str, fields: Dict[str, str]) -> str:
    text = template
    for key, value in fields.items():
        text = text.replace(f"#{key}", value)
    return text


def _extract_token_usage(result: Any) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(result, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
        )
        total_tokens = int(usage_metadata.get("total_tokens") or (prompt_tokens + completion_tokens))

    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        # Formato comum em backends OpenAI-compat
        token_usage = response_metadata.get("token_usage", {}) or {}
        if isinstance(token_usage, dict):
            prompt_tokens = int(token_usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(token_usage.get("completion_tokens") or completion_tokens)
            total_tokens = int(token_usage.get("total_tokens") or total_tokens or (prompt_tokens + completion_tokens))

        # Formato comum do Ollama em LangChain
        prompt_tokens = int(response_metadata.get("prompt_eval_count") or prompt_tokens)
        completion_tokens = int(response_metadata.get("eval_count") or completion_tokens)
        total_tokens = int(response_metadata.get("total_tokens") or total_tokens or (prompt_tokens + completion_tokens))

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": max(0, prompt_tokens),
        "completion_tokens": max(0, completion_tokens),
        "total_tokens": max(0, total_tokens),
    }


def _resolve_model_params_b(model_name: str) -> float:
    model_name_clean = str(model_name or "").strip()

    mapping = getattr(config, "MODEL_PARAMS_B_BY_MODEL", {}) or {}
    if isinstance(mapping, dict):
        model_lc = model_name_clean.lower()
        for raw_key, raw_val in mapping.items():
            key_lc = str(raw_key).strip().lower()
            if key_lc == model_lc:
                try:
                    return float(raw_val)
                except Exception:
                    break

    provider = str(getattr(config, "LLM_PROVIDER", "")).strip().lower()
    if provider == "ollama":
        params_b = resolve_model_params_b_ollama(model_name_clean)
        if params_b is not None and params_b > 0:
            return float(params_b)

    # Fallback prático: tenta inferir pelo próprio nome do modelo (ex.: ":70b", ":1.2b").
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*b\b", model_name_clean)
    if match:
        try:
            inferred = float(match.group(1))
            if inferred > 0:
                return inferred
        except Exception:
            pass

    return float(getattr(config, "GENERATION_MODEL_PARAMS_B", 0.0))


def _estimate_tflops_required(
    total_tokens: int,
    model_params_b: float,
    flops_per_param_token: float,
) -> float:
    if total_tokens <= 0 or model_params_b <= 0 or flops_per_param_token <= 0:
        return 0.0
    flops_total = total_tokens * model_params_b * 1_000_000_000.0 * flops_per_param_token
    return round(flops_total / 1_000_000_000_000.0, 6)


def _estimate_tflops_per_second(estimated_tflops_required: float, generation_seconds: float) -> float:
    if estimated_tflops_required <= 0 or generation_seconds <= 0:
        return 0.0
    return round(estimated_tflops_required / generation_seconds, 6)


def _merge_token_usage(base: Dict[str, int], extra: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": int(base.get("prompt_tokens", 0)) + int(extra.get("prompt_tokens", 0)),
        "completion_tokens": int(base.get("completion_tokens", 0)) + int(extra.get("completion_tokens", 0)),
        "total_tokens": int(base.get("total_tokens", 0)) + int(extra.get("total_tokens", 0)),
    }


def _strip_reasoning_content(raw_text: str) -> tuple[str, bool]:
    text = str(raw_text or "")
    if not text.strip():
        return "", False

    original = text

    # Remove blocos explícitos de raciocínio emitidos por modelos "thinking/reasoning".
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<thinking>.*?</thinking>", "", text)
    text = re.sub(r"(?is)<reasoning>.*?</reasoning>", "", text)
    text = re.sub(r"(?is)<analysis>.*?</analysis>", "", text)
    text = re.sub(r"(?is)<\|[^>]*thought[^>]*\|>", "", text)
    text = re.sub(r"(?is)```(?:think|thinking|reasoning|analysis|cot)\s*.*?```", "", text)
    text = text.replace("<answer>", "").replace("</answer>", "")

    first_heading = re.search(r"(?m)^###\s+\d", text)
    if first_heading:
        prefix = text[:first_heading.start()]
        if prefix.strip():
            looks_like_reasoning = re.search(
                r"(?i)\b(think|thinking|reasoning|analysis|racioc[ií]nio|pensamento|chain of thought|cot)\b",
                prefix,
            )
            # Prefixos longos antes do primeiro cabeçalho costumam ser "thinking".
            if looks_like_reasoning or len(re.findall(r"\w+", prefix, flags=re.UNICODE)) >= 40:
                text = text[first_heading.start():]

    text = text.strip()
    return text, text != original.strip()


def _section_content_word_count(text: str, sec_id: str, sec_title: str) -> int:
    header = heading(sec_id, sec_title).strip()
    clean = (text or "").strip()
    if clean.startswith(header):
        clean = clean[len(header):].strip()

    # Remove cabeçalhos de subseções e mede apenas conteúdo substantivo.
    clean = re.sub(r"(?m)^\s*###\s+\d+(?:\.\d+)*\s+[–-]\s*[^\n]*\n?", "", clean).strip()
    words = re.findall(r"\w+", clean, flags=re.UNICODE)
    return len(words)


def _is_section_text_too_short(text: str, sec_id: str, sec_title: str) -> bool:
    min_words = int(getattr(config, "SECTION_MIN_CONTENT_WORDS", 80))
    return _section_content_word_count(text, sec_id, sec_title) < min_words


def _sanitize_with_fallback(
    raw_text: str,
    sec_id: str,
    sec_title: str,
    allowed_ids: set[str],
    id2title: Dict[str, str],
    model_name: str,
) -> tuple[str, bool, bool]:
    cleaned_text, reasoning_stripped = _strip_reasoning_content(raw_text)
    preferred_input = cleaned_text if cleaned_text else str(raw_text or "")

    strict = sanitize_section_output(
        preferred_input,
        current_id=sec_id,
        allowed_ids=allowed_ids,
        id2title=id2title,
    )
    if _section_content_word_count(strict, sec_id, sec_title) > 0:
        return strict, False, reasoning_stripped

    relaxed = sanitize_section_output(
        preferred_input,
        current_id=sec_id,
        allowed_ids=None,
        id2title=id2title,
    )
    if _section_content_word_count(relaxed, sec_id, sec_title) > 0:
        print(
            f"[writer] Fallback de sanitização (modo relaxado) na seção {sec_id} para {model_name} "
            f"para evitar seção vazia."
        )
        return relaxed, True, reasoning_stripped

    fallback = preferred_input.strip() or str(raw_text or "").strip()
    if not fallback:
        return "", True, reasoning_stripped

    header = heading(sec_id, sec_title).strip()
    if not fallback.startswith(header):
        fallback = header + "\n" + fallback
    print(
        f"[writer] Fallback de conteúdo bruto na seção {sec_id} para {model_name} "
        f"porque a sanitização removeu todo o conteúdo."
    )
    return fallback, True, reasoning_stripped


def _should_retry_with_higher_cap(
    text: str,
    token_usage: Dict[str, int],
    max_tokens_used: int,
    sec_id: str,
    sec_title: str,
) -> bool:
    if max_tokens_used <= 0:
        return False
    completion_tokens = int(token_usage.get("completion_tokens", 0))
    sat_ratio = float(getattr(config, "SECTION_COMPLETION_SATURATION_RATIO", 0.97))
    hit_cap = completion_tokens >= int(max_tokens_used * sat_ratio)
    return hit_cap and _is_section_text_too_short(text, sec_id, sec_title)


def _llm_invoke_kwargs() -> Dict[str, Any]:
    provider = str(getattr(config, "LLM_PROVIDER", "")).strip().lower()
    if provider == "ollama":
        return {
            "think": bool(getattr(config, "OLLAMA_THINK", False)),
            "stream": bool(getattr(config, "OLLAMA_STREAM", False)),
        }
    return {}


def _is_transient_llm_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "remoteprotocolerror",
        "incomplete chunked read",
        "peer closed connection",
        "remotedisconnected",
        "connection reset",
        "connection aborted",
        "connection refused",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "server disconnected",
        "network is unreachable",
    )
    return any(marker in msg for marker in transient_markers)


def _invoke_llm_with_retry(
    llm: Any,
    prompt: str,
    invoke_kwargs: Dict[str, Any],
    max_retries: int,
    backoff_seconds: float,
    sec_id: str,
    model_name: str,
    max_tokens_used: int,
) -> tuple[Any, float, int]:
    retries = max(0, int(max_retries))
    attempts_total = retries + 1
    total_duration = 0.0

    for attempt in range(1, attempts_total + 1):
        with gf.stage_timer() as elapsed:
            try:
                result = llm.invoke([HumanMessage(content=prompt)], **invoke_kwargs)
            except Exception as exc:
                total_duration += float(elapsed())
                if attempt >= attempts_total or not _is_transient_llm_error(exc):
                    raise
                sleep_for = max(0.0, float(backoff_seconds)) * (2 ** (attempt - 1))
                print(
                    f"[writer] Erro transitório na seção {sec_id} ({model_name}, max_tokens={max_tokens_used}) "
                    f"tentativa {attempt}/{attempts_total}: {exc}. Retry em {sleep_for:.1f}s."
                )
                time.sleep(sleep_for)
                continue

            total_duration += float(elapsed())
            return result, total_duration, attempt

    raise RuntimeError("Falha interna inesperada no retry de chamada LLM.")


def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    writer_started = time.perf_counter()
    plan = state["plan"]
    raw_limit_top = state.get("limit_top_sections", 0)
    if raw_limit_top in (None, "", 0, "0"):
        limit_top = None
    else:
        limit_top = int(raw_limit_top)

    input_perito = state["input_perito"]
    writer_feedback = state.get("writer_feedback", []) or []
    writer_attempt = int(state.get("writer_attempt", 0)) + 1
    llm_model_name = str(state.get("llm_model_name") or getattr(config, "LLM_MODEL_NAME", "")).strip()
    benchmark_model_id = str(state.get("benchmark_model_id") or llm_model_name).strip()
    model_params_b = _resolve_model_params_b(llm_model_name)
    flops_per_param_token = float(getattr(config, "ESTIMATED_FLOPS_PER_PARAM_TOKEN", 2.0))
    templates = plan.get("guide_templates", {}) or {}
    case_fields = _extract_case_fields(input_perito)
    save_debug = bool(getattr(config, "SAVE_DEBUG_ARTIFACTS", False))

    session_dir = Path(state["session_dir"]).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Dict[str, str]] = []

    flat_all: List[Dict[str, Any]] = []
    children_map: Dict[str, List[str]] = {}
    id2title_raw: Dict[str, str] = {}

    def is_sumario(t: str) -> bool:
        t_up = normalize_title(t)
        return t_up in {"SUMÁRIO", "SUMARIO"}

    def walk(sec_list: List[Dict[str, Any]], parent_id: str | None = None) -> None:
        for i, s in enumerate(sec_list, start=1):
            if is_sumario(s.get("title", "")):
                continue
            sid = str(s.get("id") or (f"{parent_id}.{i}" if parent_id else str(i)))
            title = s["title"]
            kids = s.get("children", []) or []
            flat_all.append({"id": sid, "title": title, "children": kids})
            id2title_raw[sid] = title
            if kids:
                child_ids: List[str] = []
                for j, _ in enumerate(kids, start=1):
                    child_ids.append(f"{sid}.{j}")
                children_map[sid] = child_ids
                walk(kids, sid)

    sections_root_all = plan.get("sections", [])
    sections_root_all = [s for s in sections_root_all if not is_sumario(s.get("title", ""))]
    sections_root = sections_root_all[:limit_top] if limit_top else sections_root_all
    walk(sections_root)

    item_sections = [s for s in flat_all if "." not in s["id"]]
    item_sections.sort(key=lambda s: [int(x) for x in s["id"].split(".")])
    id2title_norm = {sid: normalize_title(tt) for sid, tt in id2title_raw.items()}

    def collect_all_descendants(parent_id: str) -> set[str]:
        stack = [parent_id]
        allowed = {parent_id}
        while stack:
            cur = stack.pop()
            for child in children_map.get(cur, []):
                allowed.add(child)
                stack.append(child)
        return allowed

    seen_tables = set()
    primary_max_tokens = int(getattr(config, "MAX_TOKENS", 3500))
    retry_max_tokens = int(getattr(config, "MAX_TOKENS_RETRY", 6000))
    llm_primary = load_llm(model_name=llm_model_name, max_tokens=primary_max_tokens)
    llm_retry = (
        load_llm(model_name=llm_model_name, max_tokens=retry_max_tokens)
        if retry_max_tokens > primary_max_tokens
        else None
    )
    invoke_kwargs = _llm_invoke_kwargs()
    invoke_max_retries = int(getattr(config, "LLM_INVOKE_MAX_RETRIES", 2))
    invoke_backoff_seconds = float(getattr(config, "LLM_INVOKE_RETRY_BACKOFF_SECONDS", 2.0))
    saturation_ratio = float(getattr(config, "SECTION_COMPLETION_SATURATION_RATIO", 0.97))
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    section_metrics: List[Dict[str, Any]] = []

    for sec in item_sections:
        sec_id = sec["id"]
        sec_title = sec["title"]
        title_up = normalize_title(sec_title)

        safe_title = re.sub(r"\W+", "_", sec_title)
        sec_dir = session_dir / f"section_{sec_id}_{safe_title}"
        sec_dir.mkdir(parents=True, exist_ok=True)

        style_constraints = (
            """Você é um modelo especializado em redação de laudos de perícia criminal da PCDF."
Siga rigorosamente o perfil de estilo e o plano de subtítulos.
Mantenha o tom técnico, impessoal e jurídico-formal; evite 1ª pessoa.
Use frases completas, terminologia técnico-pericial e linguagem precisa.
Use apenas as informações dos blocos [DESC_LOCAL], [VEST_LOCAL] e [VEST_CADAVER].
Nunca invente fatos, datas, valores ou vestígios não mencionados nesses blocos.
Se alguma informação necessária não estiver nos insumos, escreva explicitamente 'não disponível nos insumos'."""
        )
        style_constraints += (
            f"Respeite apenas a seção '{sec_id} – {sec_title.upper()}' e suas subdivisões no plano.\n"
        )

        child_outline: List[str] = []

        def add_outline(pid: str) -> None:
            for cid in children_map.get(pid, []):
                ctitle = id2title_norm.get(cid, "")
                child_outline.append(f"- {cid} – {ctitle}")
                add_outline(cid)

        add_outline(sec_id)
        child_outline_txt = "\n".join(child_outline) if child_outline else "(sem subseções no plano)"

        prompt = f"""-------------------------------- 
1) INSTRUÇÃO / TAREFA:

Você redigirá APENAS a seção "{sec_id} – {sec_title.upper()}" do laudo,
incluindo dentro do mesmo texto todas as subseções planejadas (N.N, N.N.N...) em ordem.

Plano de subseções (outline aprendido) para esta seção:
{child_outline_txt}

Perfil de estilo a ser seguido (não altere o tom):
{json.dumps(plan.get('style_profile', {}), ensure_ascii=False, indent=2)}

Regras de formatação e tom (OBRIGATÓRIAS):
{style_constraints}

Feedback de revisão para esta tentativa (aplique integralmente):
{chr(10).join(f"- {x}" for x in writer_feedback) if writer_feedback else "(primeira tentativa, sem feedback anterior)"}

Formato de saída (OBRIGATÓRIO):
{heading(sec_id, sec_title).strip()}
{{parágrafos formais de abertura do item; em seguida, escreva os subitens como cabeçalhos numerados '### <id> – <TÍTULO>' seguidos do conteúdo, respeitando o plano}}

-------------------------------- 
2) DADOS DO CASO ATUAL (EXTRAÍDOS DO SCPe_Dataset.csv):

As informações abaixo contêm APENAS os blocos permitidos:
- DESC_LOCAL
- VEST_LOCAL
- VEST_CADAVER

Use-as como base factual única; não invente nada além do que está aqui:

{input_perito}
"""
        if save_debug:
            gf.write_text(sec_dir / "prompt.txt", prompt)

        template = templates.get(title_up)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        section_invoke_attempts = 0
        section_retry_triggered = False
        section_sanitize_fallback_used = False
        section_reasoning_stripped = False
        section_primary_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        section_retry_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        allowed_ids = collect_all_descendants(sec_id)
        if template:
            rendered = _render_template(template, case_fields)
            text = f"{heading(sec_id, sec_title).strip()}\n{rendered.strip()}\n"
            duration = 0.0
            generation_source = "template"
        else:
            duration = 0.0
            result, elapsed_primary, attempts_primary = _invoke_llm_with_retry(
                llm=llm_primary,
                prompt=prompt,
                invoke_kwargs=invoke_kwargs,
                max_retries=invoke_max_retries,
                backoff_seconds=invoke_backoff_seconds,
                sec_id=sec_id,
                model_name=benchmark_model_id,
                max_tokens_used=primary_max_tokens,
            )
            duration += elapsed_primary
            section_invoke_attempts += attempts_primary
            text = result.content or ""
            token_usage = _extract_token_usage(result)
            section_primary_usage = dict(token_usage)
            generation_source = "llm" if attempts_primary == 1 else "llm_retry_conn"

            text, fallback_used, reasoning_stripped = _sanitize_with_fallback(
                raw_text=text,
                sec_id=sec_id,
                sec_title=sec_title,
                allowed_ids=allowed_ids,
                id2title=id2title_norm,
                model_name=benchmark_model_id,
            )
            section_sanitize_fallback_used = section_sanitize_fallback_used or fallback_used
            section_reasoning_stripped = section_reasoning_stripped or reasoning_stripped
            header = heading(sec_id, sec_title).strip()
            if not text.strip().startswith(header):
                text = header + "\n" + text.lstrip()

            should_retry = llm_retry is not None and _should_retry_with_higher_cap(
                text=text,
                token_usage=token_usage,
                max_tokens_used=primary_max_tokens,
                sec_id=sec_id,
                sec_title=sec_title,
            )
            if should_retry:
                section_retry_triggered = True
                print(
                    f"[writer] Retry seção {sec_id} com limite maior "
                    f"({primary_max_tokens} -> {retry_max_tokens}) para {benchmark_model_id}"
                )
                result_retry, elapsed_retry, attempts_retry = _invoke_llm_with_retry(
                    llm=llm_retry,
                    prompt=prompt,
                    invoke_kwargs=invoke_kwargs,
                    max_retries=invoke_max_retries,
                    backoff_seconds=invoke_backoff_seconds,
                    sec_id=sec_id,
                    model_name=benchmark_model_id,
                    max_tokens_used=retry_max_tokens,
                )
                duration += elapsed_retry
                section_invoke_attempts += attempts_retry
                retry_text = result_retry.content or ""
                retry_usage = _extract_token_usage(result_retry)
                section_retry_usage = dict(retry_usage)
                token_usage = _merge_token_usage(token_usage, retry_usage)
                text, fallback_used, reasoning_stripped = _sanitize_with_fallback(
                    raw_text=retry_text,
                    sec_id=sec_id,
                    sec_title=sec_title,
                    allowed_ids=allowed_ids,
                    id2title=id2title_norm,
                    model_name=benchmark_model_id,
                )
                section_sanitize_fallback_used = section_sanitize_fallback_used or fallback_used
                section_reasoning_stripped = section_reasoning_stripped or reasoning_stripped
                generation_source = "llm_retry"

        prompt_tokens_total += token_usage["prompt_tokens"]
        completion_tokens_total += token_usage["completion_tokens"]
        total_tokens_total += token_usage["total_tokens"]

        if template:
            text, fallback_used, reasoning_stripped = _sanitize_with_fallback(
                raw_text=text,
                sec_id=sec_id,
                sec_title=sec_title,
                allowed_ids=allowed_ids,
                id2title=id2title_norm,
                model_name=benchmark_model_id,
            )
            section_sanitize_fallback_used = section_sanitize_fallback_used or fallback_used
            section_reasoning_stripped = section_reasoning_stripped or reasoning_stripped

        tables = re.findall(r"\n\|.+\|\n\|[-| ]+\|\n(?:\|.*\|\n)+", text)
        for tb in tables:
            h = hashlib.md5(tb.encode("utf-8")).hexdigest()
            if h in seen_tables:
                text = text.replace(tb, "\n> Tabela similar já apresentada anteriormente; não reproduzida.\n")
            else:
                seen_tables.add(h)

        header = heading(sec_id, sec_title).strip()
        if not text.strip().startswith(header):
            text = header + "\n" + text.lstrip()

        gf.write_text(sec_dir / "output.txt", text)
        if save_debug:
            gf.write_json(sec_dir / "timing.json", {"duration_seconds": duration})
        produced.append({"id": sec_id, "title": sec_title, "text": text})

        completion_sat_primary = int(section_primary_usage.get("completion_tokens", 0)) >= int(
            primary_max_tokens * saturation_ratio
        )
        completion_sat_retry = int(section_retry_usage.get("completion_tokens", 0)) >= int(
            retry_max_tokens * saturation_ratio
        ) if section_retry_triggered and retry_max_tokens > 0 else False
        truncation_suspected = (
            (completion_sat_primary or completion_sat_retry) and _is_section_text_too_short(text, sec_id, sec_title)
        )

        section_metrics.append(
            {
                "id": sec_id,
                "title": sec_title,
                "duration_seconds": round(float(duration), 3),
                "source": generation_source,
                "invoke_attempts": section_invoke_attempts,
                "retry_triggered": section_retry_triggered,
                "sanitize_fallback_used": section_sanitize_fallback_used,
                "reasoning_content_stripped": section_reasoning_stripped,
                "completion_saturated_primary": completion_sat_primary,
                "completion_saturated_retry": completion_sat_retry,
                "truncation_suspected": truncation_suspected,
                "max_tokens_primary": primary_max_tokens,
                "max_tokens_retry": retry_max_tokens if section_retry_triggered else None,
                "token_usage": token_usage,
                "token_usage_primary": section_primary_usage,
                "token_usage_retry": section_retry_usage if section_retry_triggered else None,
                "estimated_tflops_required": _estimate_tflops_required(
                    token_usage.get("total_tokens", 0),
                    model_params_b=model_params_b,
                    flops_per_param_token=flops_per_param_token,
                ),
            }
        )

    generation_seconds = round(float(time.perf_counter() - writer_started), 3)
    estimated_tflops_required = _estimate_tflops_required(
        total_tokens_total,
        model_params_b=model_params_b,
        flops_per_param_token=flops_per_param_token,
    )
    estimated_tflops_per_second = _estimate_tflops_per_second(
        estimated_tflops_required,
        generation_seconds,
    )

    sections_with_retry = [s["id"] for s in section_metrics if s.get("retry_triggered")]
    sections_with_saturation = [
        s["id"]
        for s in section_metrics
        if s.get("completion_saturated_primary") or s.get("completion_saturated_retry")
    ]
    sections_with_truncation = [s["id"] for s in section_metrics if s.get("truncation_suspected")]
    sections_with_sanitize_fallback = [s["id"] for s in section_metrics if s.get("sanitize_fallback_used")]
    sections_with_reasoning_stripped = [s["id"] for s in section_metrics if s.get("reasoning_content_stripped")]
    generation_conditions = {
        "had_retry": bool(sections_with_retry),
        "had_completion_saturation": bool(sections_with_saturation),
        "had_truncation_suspected": bool(sections_with_truncation),
        "had_sanitize_fallback": bool(sections_with_sanitize_fallback),
        "had_reasoning_content_stripped": bool(sections_with_reasoning_stripped),
        "sections_with_retry": sections_with_retry,
        "sections_with_completion_saturation": sections_with_saturation,
        "sections_with_truncation_suspected": sections_with_truncation,
        "sections_with_sanitize_fallback": sections_with_sanitize_fallback,
        "sections_with_reasoning_content_stripped": sections_with_reasoning_stripped,
        "completion_saturation_ratio": saturation_ratio,
        "ollama_stream_enabled": bool(invoke_kwargs.get("stream", True)),
        "ollama_think_enabled": bool(invoke_kwargs.get("think", False)),
    }

    generation_perf = {
        "generation_seconds": generation_seconds,
        "llm_provider": getattr(config, "LLM_PROVIDER", ""),
        "llm_model": llm_model_name,
        "benchmark_model_id": benchmark_model_id,
        "token_usage": {
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens_total,
            "total_tokens": total_tokens_total,
        },
        "estimated_tflops_required": estimated_tflops_required,
        "estimated_tflops_per_second": estimated_tflops_per_second,
        "estimated_theoretical_tflops_required": estimated_tflops_required,
        "estimated_theoretical_tflops_per_second": estimated_tflops_per_second,
        "tflops_formula": "total_tokens * model_params_b * 1e9 * flops_per_param_token / 1e12",
        "tflops_formula_per_second": "estimated_tflops_required / generation_seconds",
        "theoretical_tflops_formula": "total_tokens * model_params_b * 1e9 * flops_per_param_token / 1e12",
        "theoretical_tflops_formula_per_second": "estimated_theoretical_tflops_required / generation_seconds",
        "tflops_params": {
            "model_params_b": model_params_b,
            "flops_per_param_token": flops_per_param_token,
        },
        "generation_conditions": generation_conditions,
        "sections": section_metrics,
    }
    generation_metrics = {
        "dataset_row_index": state.get("dataset_row_index"),
        "llm_provider": getattr(config, "LLM_PROVIDER", ""),
        "llm_model_name": llm_model_name,
        "benchmark_model_id": benchmark_model_id,
        "textual_quality": {"score": None, "status": "pending_evaluation"},
        "dataset_adherence": {"score": None, "status": "pending_evaluation"},
        "generation_token_usage": generation_perf["token_usage"],
        "estimated_tflops_required": estimated_tflops_required,
        "estimated_tflops_per_second": estimated_tflops_per_second,
        "estimated_theoretical_tflops_required": estimated_tflops_required,
        "estimated_theoretical_tflops_per_second": estimated_tflops_per_second,
        "generation_seconds": generation_seconds,
        "generation_conditions": generation_conditions,
        "generation_perf": generation_perf,
        "tflops_formula": "total_tokens * model_params_b * 1e9 * flops_per_param_token / 1e12",
        "tflops_formula_per_second": "estimated_tflops_required / generation_seconds",
        "theoretical_tflops_formula": "total_tokens * model_params_b * 1e9 * flops_per_param_token / 1e12",
        "theoretical_tflops_formula_per_second": "estimated_theoretical_tflops_required / generation_seconds",
        "tflops_params": {
            "model_params_b": model_params_b,
            "flops_per_param_token": flops_per_param_token,
        },
    }
    laudo_gerado = "\n".join((s.get("text", "").strip() for s in produced if s.get("text"))).strip()
    if laudo_gerado:
        laudo_gerado += "\n"

    return {
        "sections_texts": produced,
        "laudo_gerado": laudo_gerado,
        "plan": plan,
        "limit_top_sections": raw_limit_top if raw_limit_top not in (None, "") else 0,
        "session_dir": str(session_dir),
        "writer_attempt": writer_attempt,
        "writer_feedback": [],
        "llm_model_name": llm_model_name,
        "benchmark_model_id": benchmark_model_id,
        "generation_perf": generation_perf,
        "generation_metrics": generation_metrics,
    }
