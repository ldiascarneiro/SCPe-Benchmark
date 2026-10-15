# Created by Leandro Carneiro at 18/09/2025
# Description: 
# -------------------------------------------------------------
# services_packages/generation_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json, re
import csv

import config
from services_packages import general_functions as gf

# ---------------- Sessão / IO ----------------
class SessionManager:
    def __init__(self):
        self.session_dir: Path | None = None
        self.exec_log_path: Path | None = None

    def start(self, suffix: str | None = None):
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        base_name = f"generation_{gf.timestamp()}"
        if suffix:
            suffix_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix)).strip("_")
            if suffix_safe:
                base_name = f"{base_name}_{suffix_safe}"

        candidate = (config.OUTPUT_DIR / base_name).resolve()
        counter = 1
        while candidate.exists():
            candidate = (config.OUTPUT_DIR / f"{base_name}_{counter:02d}").resolve()
            counter += 1

        self.session_dir = candidate
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.exec_log_path = self.session_dir / "execution_log.jsonl"

    def save_text(self, relpath: str, content: str) -> Path:
        assert self.session_dir is not None, "session_dir não inicializado"
        p = self.session_dir / relpath
        gf.write_text(p, content)
        return p

    def save_json(self, relpath: str, data: dict) -> Path:
        assert self.session_dir is not None, "session_dir não inicializado"
        p = self.session_dir / relpath
        gf.write_json(p, data)
        return p

    def log(self, payload: Dict[str, Any]):
        assert self.exec_log_path is not None
        payload = {**payload, "ts": datetime.now().isoformat(timespec="seconds")}
        with open(self.exec_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def load_case_from_csv(sess: SessionManager, row_index: int | None = None) -> Dict[str, Any]:
    """
    Carrega um único caso do SCPe_Dataset.csv usando delimitador '#'
    e retorna apenas os campos permitidos.
    """
    csv_path = Path(getattr(config, "DATASET_CSV_PATH", ""))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    delimiter = getattr(config, "CSV_DELIMITER", "#")
    lock_first_useful = bool(getattr(config, "LOCK_TO_FIRST_USEFUL_ROW", True))

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV vazio: {csv_path}")
    required = ("DESC_LOCAL", "VEST_LOCAL", "VEST_CADAVER")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes no CSV: {missing}")

    def is_useful(r: Dict[str, Any]) -> bool:
        return any((r.get(c) or "").strip() for c in required)

    if row_index is not None:
        idx = int(row_index)
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"DATASET_ROW_INDEX={idx} fora do intervalo [0, {len(rows)-1}]")
    elif lock_first_useful:
        idx = None
        for i, r in enumerate(rows):
            if is_useful(r):
                idx = i
                break
        if idx is None:
            raise RuntimeError("Nenhuma linha útil encontrada no CSV para DESC_LOCAL/VEST_LOCAL/VEST_CADAVER.")
    else:
        idx = int(getattr(config, "DATASET_ROW_INDEX", 0))
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"DATASET_ROW_INDEX={idx} fora do intervalo [0, {len(rows)-1}]")

    row = rows[idx]

    desc_local = (row.get("DESC_LOCAL") or "").strip()
    vest_local = (row.get("VEST_LOCAL") or "").strip()
    vest_cadaver = (row.get("VEST_CADAVER") or "").strip()
    input_perito = (
        "CASO EXTRAÍDO EXCLUSIVAMENTE DO SCPe_Dataset.csv\n"
        f"Linha do dataset: {idx}\n"
        "Colunas utilizadas: DESC_LOCAL, VEST_LOCAL, VEST_CADAVER\n\n"
        "[DESC_LOCAL]\n"
        f"{desc_local or 'não disponível'}\n\n"
        "[VEST_LOCAL]\n"
        f"{vest_local or 'não disponível'}\n\n"
        "[VEST_CADAVER]\n"
        f"{vest_cadaver or 'não disponível'}\n"
    )

    # Persistência explícita do caso usado
    sess.save_json("00_dataset_case.json", {
        "row_index": idx,
        "lock_to_first_useful_row": lock_first_useful,
        "DESC_LOCAL": desc_local,
        "VEST_LOCAL": vest_local,
        "VEST_CADAVER": vest_cadaver,
    })
    sess.save_text("00_input_perito.txt", input_perito)

    return {
        "input_perito": input_perito,
        "docs_complementares": [],
        "guia_conteudo": "",
        "case_metadata": {"row_index": idx},
    }

# ---------------- Normalizações / Parsing ----------------
_H_RE = re.compile(r"^(?P<num>(\d+(\.\d+){0,2}))?\s*(?P<title>[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ][\w\s\-/()ºª]+)$")

def normalize_title(t: str) -> str:
    t = re.sub(r"\s+\d+$", "", t.strip())
    t_up = re.sub(r"\s+", " ", t.upper().strip(" -–\t"))
    for canon, syns in getattr(config, "SECTION_SYNONYMS", {}).items():
        if t_up == canon or t_up in [s.upper() for s in syns]:
            t_up = canon
            break
    return t_up

def is_sumario(title: str) -> bool:
    t = normalize_title(title)
    return t in {"SUMÁRIO", "SUMARIO"}

def extract_headings(text: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    found = []
    max_depth = getattr(config, "MAX_OUTLINE_DEPTH", 3)
    for l in lines:
        m = _H_RE.match(l)
        if m:
            num = m.group("num")
            title = m.group("title").strip()
            depth = 1 + (num.count(".") if num else 0)
            if depth <= max_depth:
                found.append({"raw": l, "num": num, "title": title, "depth": depth})
    return found

def outline_tree_from_doc(doc: str) -> Dict[str, Any]:
    hs = extract_headings(doc)
    hs = [h for h in hs if not is_sumario(h["title"])]
    root = {"title": "__ROOT__", "children": []}
    stack = [(0, root)]
    for h in hs:
        node = {"title": normalize_title(h["title"]), "children": []}
        d = h["depth"]
        while stack and stack[-1][0] >= d:
            stack.pop()
        stack[-1][1]["children"].append(node)
        stack.append((d, node))
    return root

def walk_paths(node: Dict[str, Any], prefix: Tuple[str, ...], acc: List[Tuple[Tuple[str, ...], int]]):
    for child in node.get("children", []):
        path = prefix + (child["title"],)
        acc.append((path, len(acc)))
        walk_paths(child, path, acc)

def induce_outline_from_gold(docs: List[str]) -> Dict[str, Any]:
    trees = [outline_tree_from_doc(d) for d in docs]
    freq: Dict[Tuple[str, ...], Dict[str, float]] = {}
    orders: Dict[Tuple[str, ...], List[int]] = {}
    for t in trees:
        paths: List[Tuple[Tuple[str, ...], int]] = []
        walk_paths(t, tuple(), paths)
        seen = set()
        for path, idx in paths:
            if path in seen:
                continue
            seen.add(path)
            meta = freq.setdefault(path, {"count": 0})
            meta["count"] += 1
            orders.setdefault(path, []).append(idx)

    n_docs = max(1, len(trees))
    support_min = getattr(config, "STRUCTURE_MIN_SUPPORT", 0.5)

    def build_children(prefix: Tuple[str, ...]) -> List[Dict[str, Any]]:
        candidates = [p for p in freq.keys() if len(p) == len(prefix) + 1 and p[:len(prefix)] == prefix]
        kept = []
        for p in candidates:
            support = freq[p]["count"] / n_docs
            if support >= support_min:
                kept.append((p, sum(orders[p]) / len(orders[p])))
        kept.sort(key=lambda x: x[1])
        out = []
        for p, _ in kept:
            out.append({"title": p[-1], "children": build_children(p)})
        return out

    sections = build_children(tuple())
    return {"sections": sections}


def parse_guide_structure(guia: str) -> Dict[str, Any]:
    """
    Extrai a estrutura hierárquica do guia:
      - seções principais: N. TÍTULO
      - subseções: N.M / N.M.K
    """
    lines = guia.splitlines()
    sections: List[Dict[str, Any]] = []
    required_subsections: Dict[str, List[Dict[str, str]]] = {}
    in_sections_block = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if normalize_title(line).startswith("SEÇÕES DO DOCUMENTO"):
            in_sections_block = True
            continue

        if not in_sections_block:
            continue

        m_top = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m_top:
            sid = m_top.group(1)
            title = normalize_title(m_top.group(2))
            sections.append({"id": sid, "title": title, "children": []})
            required_subsections.setdefault(sid, [])
            continue

        m_sub = re.match(r"^(\d+(?:\.\d+)+)\s+(.+)$", line)
        if m_sub:
            sub_id = m_sub.group(1)
            sub_title = normalize_title(m_sub.group(2))
            parent = sub_id.split(".")[0]
            required_subsections.setdefault(parent, []).append({"id": sub_id, "title": sub_title})
            continue

    return {"sections": sections, "required_subsections": required_subsections}


def extract_template_rules(guia: str) -> Dict[str, str]:
    """
    Extrai templates literais declarados no guia após linhas com "Modelo:".
    O mapeamento é feito pelo título canônico da seção principal.
    """
    lines = guia.splitlines()
    templates: Dict[str, str] = {}
    current_section_title: str | None = None

    for i, raw in enumerate(lines):
        line = raw.strip()
        m_top = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m_top:
            current_section_title = normalize_title(m_top.group(2))
            continue

        if current_section_title and "MODELO:" in normalize_title(line):
            for j in range(i + 1, min(i + 8, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                if candidate.startswith('"') and candidate.endswith('"') and len(candidate) > 2:
                    templates[current_section_title] = candidate[1:-1].strip()
                    break
    return templates


def build_plan_from_guide(
    guide_struct: Dict[str, Any],
    learned_outline: Dict[str, Any],
    strict_order: bool = True,
    fallback_to_gold: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Monta o plano final de seções priorizando o guia de conteúdo.
    """
    learned_sections = learned_outline.get("sections", []) or []
    by_title: Dict[str, Dict[str, Any]] = {}
    for sec in learned_sections:
        by_title[normalize_title(sec.get("title", ""))] = sec

    plan_sections: List[Dict[str, Any]] = []
    trace_sections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    used_titles = set()

    guide_sections = guide_struct.get("sections", []) or []
    if not guide_sections:
        warnings.append("Guia sem estrutura parseável; aplicando fallback integral para padrão-ouro.")
        for i, sec in enumerate(learned_sections, start=1):
            title = normalize_title(sec.get("title", ""))
            sid = str(i)
            plan_sections.append({"id": sid, "title": title, "children": sec.get("children", []) or []})
            trace_sections.append({"id": sid, "title": title, "origin": "gold_only"})
        return (
            {"sections": plan_sections},
            {"sections": trace_sections},
            warnings,
        )

    for gsec in guide_sections:
        gid = str(gsec.get("id", "")).strip()
        gtitle = normalize_title(gsec.get("title", ""))
        learned = by_title.get(gtitle)

        if learned:
            children = learned.get("children", []) or []
            origin = "guide+gold"
            used_titles.add(gtitle)
        else:
            children = []
            origin = "guide_only"
            warnings.append(f"Seção '{gid} – {gtitle}' não encontrada no padrão-ouro; criada vazia a partir do guia.")

        plan_sections.append({"id": gid, "title": gtitle, "children": children})
        trace_sections.append({"id": gid, "title": gtitle, "origin": origin})

    if not strict_order and fallback_to_gold:
        for sec in learned_sections:
            title = normalize_title(sec.get("title", ""))
            if title in used_titles:
                continue
            next_id = str(len(plan_sections))
            plan_sections.append({"id": next_id, "title": title, "children": sec.get("children", []) or []})
            trace_sections.append({"id": next_id, "title": title, "origin": "gold_fallback"})
            warnings.append(f"Seção extra do padrão-ouro incluída via fallback: '{title}'.")

    return (
        {"sections": plan_sections},
        {"sections": trace_sections},
        warnings,
    )

def build_style_profile(docs: List[str]) -> Dict[str, Any]:
    common_terms = set()
    for d in docs:
        for token in re.findall(r"[A-Za-zÀ-ÿ\-]{5,}", d):
            if token.isupper():
                common_terms.add(token)
    return {
        "tone": "técnico, impessoal, jurídico-formal",
        "format_conventions": [
            "títulos em MAIÚSCULAS com numeração hierárquica",
            "referir fotografias/figuras/tabelas quando existirem; não inventar numeração",
            "evitar 1ª pessoa e linguagem coloquial",
        ],
        "terminology": sorted(list(common_terms))[:80],
    }

def extract_section_guidance(guia_conteudo: str, section_title: str) -> str:
    if not guia_conteudo.strip():
        return ""
    title_norm = normalize_title(section_title)
    lines = guia_conteudo.splitlines()
    start = None
    for i, ln in enumerate(lines):
        ln_norm = normalize_title(ln.strip(":#- "))
        if ln_norm == title_norm:
            start = i
            break
    if start is None:
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith(title_norm):
                start = i
                break
    if start is None:
        return guia_conteudo[:1500]
    block = [lines[start]]
    for j in range(start + 1, len(lines)):
        nxt = lines[j]
        if re.match(r"^[A-ZÁÉÍÓÚÃÕÇ0-9][A-ZÁÉÍÓÚÃÕÇa-z0-9 \-_/().]*$", nxt.strip()) and nxt.strip() == nxt.strip().upper():
            break
        if nxt.strip().startswith(("###", "##", "#")):
            break
        block.append(nxt)
    text = "\n".join(block).strip()
    return text[:3000]

def heading(sid: str, title: str) -> str:
    title_clean = re.sub(r"\s+\d+$", "", title).upper()
    return f"### {sid} – {title_clean}\n\n"

def sanitize_section_output(
    text: str,
    current_id: str,
    allowed_ids: set[str] | None = None,
    id2title: dict[str, str] | None = None
) -> str:
    _re = re
    text = _re.sub(
        r'^\s*####+\s*([0-9]+(?:\.[0-9]+)*)\s*(?:[-–]\s*)?([^\n]*)$',
        lambda m: f"### {m.group(1)} – {m.group(2).upper().strip()}",
        text, flags=_re.MULTILINE
    )
    text = _re.sub(
        r'^\s*([0-9]+(?:\.[0-9]+)*)\s+([A-ZÁÉÍÓÚÃÕÇa-z0-9][^\n]*)$',
        lambda m: f"### {m.group(1)} – {m.group(2).upper().strip()}",
        text, flags=_re.MULTILINE
    )
    text = _re.sub(
        rf'^###\s+{_re.escape(current_id)}\s+([–-])\s*(.*)$',
        lambda m: f"### {current_id} – {m.group(2).upper().strip()}",
        text, flags=_re.MULTILINE
    )
    h3 = _re.compile(r'^###\s+([0-9]+(?:\.[0-9]+)*)\s+–\s*([^\n]*)\n?', _re.MULTILINE)
    starts = [m.start() for m in h3.finditer(text)]
    if not starts:
        return text
    starts.append(len(text))
    blocks = []
    for i in range(len(starts)-1):
        a, b = starts[i], starts[i+1]
        head = text[a:b].splitlines()[0]
        m = h3.search(head)
        if not m:
            continue
        sid = m.group(1)
        title = m.group(2).strip().upper()
        blocks.append((sid, title, a, b))

    scoped = []
    for sid, title, a, b in blocks:
        if not (sid == current_id or sid.startswith(current_id + ".")):
            continue
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        scoped.append((sid, title, a, b))

    out_chunks = []
    seen = set()
    by_id: dict[str, list[tuple[str, str, int, int]]] = {}
    for item in scoped:
        by_id.setdefault(item[0], []).append(item)

    for sid in sorted(by_id.keys(), key=lambda x: [int(n) for n in x.split(".")]):
        if sid in seen:
            continue
        group = by_id[sid]
        expected = (id2title or {}).get(sid)
        chosen = None
        if expected:
            for cand in group:
                if cand[1] == expected:
                    chosen = cand
                    break
        chosen = chosen or group[0]
        _, _, a, b = chosen
        chunk = text[a:b]
        if expected:
            lines = chunk.splitlines(True)
            if lines:
                lines[0] = f"### {sid} – {expected}\n"
                chunk = "".join(lines)
        out_chunks.append(chunk)
        seen.add(sid)

    result = "".join(out_chunks)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result
