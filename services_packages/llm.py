# Created by Leandro Carneiro at 18/09/2025
# Description: 
# -------------------------------------------------------------
# services_packages/llm.py
from typing import Any, Iterable
import json
import re
import time
from urllib import error, request
import config

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

_OLLAMA_MODEL_PARAMS_CACHE_B: dict[str, float] = {}
_OLLAMA_CHAT_MODEL_CACHE: dict[tuple[str, str, int, int, float], ChatOllama] = {}


def _get_openai_api_key() -> str | None:
    key = getattr(config, "OPENAI_API_KEY", None)
    if key is None:
        return None
    key_str = str(key).strip()
    return key_str or None


def _normalize_ollama_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


def _ollama_request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_s: int = 60,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="ignore").strip()

    if not body:
        return {}

    # Alguns endpoints retornam JSONL; pegamos o último objeto válido.
    last_obj = None
    for line in body.splitlines():
        clean = line.strip()
        if not clean:
            continue
        try:
            last_obj = json.loads(clean)
        except json.JSONDecodeError:
            continue
    if last_obj is not None:
        return last_obj

    return json.loads(body)


def _list_ollama_models(base_url: str) -> set[str]:
    payload = _ollama_request_json("GET", f"{base_url}/api/tags", timeout_s=120)
    installed: set[str] = set()

    if isinstance(payload, dict):
        models = payload.get("models")
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, dict):
                    continue
                for key in ("name", "model"):
                    value = item.get(key)
                    if value:
                        installed.add(str(value).strip())
    return installed


def _parse_params_b(raw_value: Any) -> float | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value <= 0:
            return None
        # Quando vier contagem absoluta de parâmetros.
        if value > 1_000_000:
            return value / 1_000_000_000.0
        return value

    text = str(raw_value).strip()
    if not text:
        return None

    # Ex.: "70B", "1.2B", "671M", "70000000000"
    m = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*([BMK])?\b", text)
    if not m:
        return None

    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    if suffix == "B":
        return num
    if suffix == "M":
        return num / 1000.0
    if suffix == "K":
        return num / 1_000_000.0

    # Sem sufixo; pode ser absoluto (ex.: 70000000000)
    if num > 1_000_000:
        return num / 1_000_000_000.0
    return num


def _resolve_params_b_from_ollama_tags(base_url: str, model_name: str) -> float | None:
    payload = _ollama_request_json("GET", f"{base_url}/api/tags", timeout_s=120)
    if not isinstance(payload, dict):
        return None

    models = payload.get("models")
    if not isinstance(models, list):
        return None

    wanted = str(model_name).strip().lower()
    for item in models:
        if not isinstance(item, dict):
            continue
        candidates = [str(item.get("name") or "").strip(), str(item.get("model") or "").strip()]
        if all(not c for c in candidates):
            continue
        if wanted not in [c.lower() for c in candidates]:
            continue

        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        for key in ("parameter_size", "parameters"):
            val = _parse_params_b(details.get(key))
            if val and val > 0:
                return val
    return None


def _resolve_params_b_from_ollama_show(base_url: str, model_name: str) -> float | None:
    payload = _ollama_request_json(
        "POST",
        f"{base_url}/api/show",
        payload={"name": model_name},
        timeout_s=120,
    )
    if not isinstance(payload, dict):
        return None

    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    for key in ("parameter_size", "parameters"):
        val = _parse_params_b(details.get(key))
        if val and val > 0:
            return val

    model_info = payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
    for key in ("general.parameter_count", "parameter_count", "params_count"):
        val = _parse_params_b(model_info.get(key))
        if val and val > 0:
            return val

    return None


def _pull_ollama_model(base_url: str, model_name: str) -> None:
    _ollama_request_json(
        "POST",
        f"{base_url}/api/pull",
        payload={"name": model_name, "stream": False},
        timeout_s=60 * 60,
    )


def _is_transient_pull_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "remote end closed connection",
        "remotedisconnected",
        "peer closed connection",
        "connection reset",
        "connection aborted",
        "connection refused",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "bad gateway",
        "service unavailable",
        "incomplete chunked read",
        "network is unreachable",
    )
    return any(marker in msg for marker in transient_markers)


def _pull_ollama_model_with_retry(base_url: str, model_name: str) -> None:
    retries = max(0, int(getattr(config, "OLLAMA_PULL_MAX_RETRIES", 3)))
    backoff = max(0.0, float(getattr(config, "OLLAMA_PULL_RETRY_BACKOFF_SECONDS", 4.0)))
    attempts_total = retries + 1

    for attempt in range(1, attempts_total + 1):
        try:
            _pull_ollama_model(base_url, model_name)
            return
        except Exception as exc:
            # Às vezes a conexão cai no fim do pull; confirma por tags.
            try:
                installed = _list_ollama_models(base_url)
                if model_name in installed:
                    return
            except Exception:
                pass

            if attempt >= attempts_total or not _is_transient_pull_error(exc):
                raise

            sleep_for = backoff * (2 ** (attempt - 1))
            print(
                f"[OLLAMA] Falha transitória ao baixar {model_name} "
                f"(tentativa {attempt}/{attempts_total}): {exc}. Retry em {sleep_for:.1f}s."
            )
            time.sleep(sleep_for)


def ensure_ollama_models_available(model_names: Iterable[str]) -> None:
    base_url = _normalize_ollama_base_url(getattr(config, "CHAT_MODEL_URL", ""))
    if not base_url:
        raise RuntimeError("CHAT_MODEL_URL não configurado para o provider 'ollama'.")

    requested = [str(m).strip() for m in model_names if str(m).strip()]
    if not requested:
        return

    try:
        installed = _list_ollama_models(base_url)
    except error.URLError as exc:
        raise RuntimeError(f"Falha ao conectar ao Ollama em {base_url}: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Falha ao consultar modelos no Ollama: {exc}") from exc

    for model_name in requested:
        if model_name in installed:
            print(f"[OLLAMA] Modelo já disponível: {model_name}")
            continue

        print(f"[OLLAMA] Baixando modelo: {model_name} ...")
        try:
            _pull_ollama_model_with_retry(base_url, model_name)
        except error.URLError as exc:
            raise RuntimeError(f"Falha ao baixar modelo '{model_name}': {exc}") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Falha ao baixar modelo '{model_name}': {exc}") from exc
        installed.add(model_name)
        print(f"[OLLAMA] Download concluído: {model_name}")


def resolve_model_params_b_ollama(model_name: str) -> float | None:
    model = str(model_name or "").strip()
    if not model:
        return None

    cache_key = model.lower()
    if cache_key in _OLLAMA_MODEL_PARAMS_CACHE_B:
        return _OLLAMA_MODEL_PARAMS_CACHE_B[cache_key]

    base_url = _normalize_ollama_base_url(getattr(config, "CHAT_MODEL_URL", ""))
    if not base_url:
        return None

    try:
        params_b = _resolve_params_b_from_ollama_tags(base_url, model)
        if params_b is None:
            params_b = _resolve_params_b_from_ollama_show(base_url, model)
    except Exception:  # pylint: disable=broad-except
        return None

    if params_b is not None and params_b > 0:
        _OLLAMA_MODEL_PARAMS_CACHE_B[cache_key] = float(params_b)
        return float(params_b)
    return None


def load_llm(model_name: str | None = None, max_tokens: int | None = None) -> Any:
    provider = str(getattr(config, "LLM_PROVIDER", "openai")).strip().lower()
    resolved_model_name = str(model_name or getattr(config, "LLM_MODEL_NAME", "")).strip()
    resolved_max_tokens = int(max_tokens if max_tokens is not None else getattr(config, "MAX_TOKENS", 20000))

    if provider == "ollama":
        base_url = str(getattr(config, "CHAT_MODEL_URL", "")).strip()
        num_ctx = int(getattr(config, "LLM_MODEL_NUM_CTX", 16384))
        temperature = 0.1
        cache_key = (base_url, resolved_model_name, num_ctx, resolved_max_tokens, float(temperature))
        cached = _OLLAMA_CHAT_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model = ChatOllama(
            model=resolved_model_name,
            base_url=base_url,
            num_ctx=num_ctx,
            num_predict=resolved_max_tokens,
            temperature=temperature,
        )
        _OLLAMA_CHAT_MODEL_CACHE[cache_key] = model
        return model

    if provider == "local_vllm":
        return ChatOpenAI(
            model=resolved_model_name,
            base_url=config.CHAT_MODEL_URL,
            api_key="dummy-key",  # vLLM ignora, mas LangChain exige
            max_tokens=resolved_max_tokens
        )

    return ChatOpenAI(
        model=resolved_model_name,
        temperature=0.1,
        api_key=_get_openai_api_key(),
        max_tokens=resolved_max_tokens,
    )
