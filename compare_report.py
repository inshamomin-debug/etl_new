import os
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Tuple, List

INPUT_TEXT_PATH = os.getenv("INPUT_TEXT_PATH", "data/input.txt")
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", "output/result.json")
COMPARISON_JSON_PATH = os.getenv("COMPARISON_JSON_PATH", "output/reports/comparison_report.json")
COMPARISON_MD_PATH = os.getenv("COMPARISON_MD_PATH", "output/reports/comparison_report.md")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.update(flatten_json(v, p))
    else:
        out[prefix] = obj
    return out


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").lower()
    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    n = normalize_text(s)
    if not n:
        return []
    return [t for t in n.split(" ") if len(t) > 1]


def find_snippet(text: str, term: str, context: int = 80) -> Tuple[bool, str]:
    if term is None:
        return False, ""
    raw_term = str(term).strip()
    if raw_term == "":
        return False, ""
    idx = text.lower().find(raw_term.lower())
    if idx != -1:
        start = max(0, idx - context)
        end = min(len(text), idx + len(raw_term) + context)
        snippet = text[start:end].replace("\n", " ")
        return True, snippet
    tokens = tokenize(raw_term)
    for t in tokens:
        idx2 = text.lower().find(t.lower())
        if idx2 != -1:
            start = max(0, idx2 - context)
            end = min(len(text), idx2 + len(t) + context)
            snippet = text[start:end].replace("\n", " ")
            return True, snippet
    return False, ""


def extract_numbers_from_text(text: str) -> List[float]:
    nums = []
    for m in re.finditer(r"(?<!\w)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)(?!\w)", text):
        s = m.group(0).replace(",", "")
        try:
            nums.append(float(s))
        except Exception:
            continue
    return nums


def numeric_matches(value: Any, text: str) -> Tuple[bool, str]:
    """
    If value looks numeric, check if matching number exists in text.
    Returns (matched, evidence_string)
    """
    try:
        # try direct numeric conversion
        if isinstance(value, (int, float)):
            valf = float(value)
        else:
            s = str(value).strip().replace(",", "")
            # if value contains non-digit words like "6 lakh", try to extract digits
            num_match = re.search(r"\d[\d,\.]*", s)
            if num_match:
                valf = float(num_match.group(0).replace(",", ""))
            else:
                return False, "no numeric literal found in JSON value"
    except Exception:
        return False, "value not numeric"

    text_nums = extract_numbers_from_text(text)
    if not text_nums:
        return False, "no numeric literals found in text"

    # check exact or approximate match
    for tn in text_nums:
        if abs(tn - valf) < 1e-6 or (valf != 0 and abs(tn - valf) / max(abs(valf), 1) < 0.01):
            return True, f"matched numeric {valf} in text (found {tn})"
    # also check for common conversions (lakhs -> multiply)
    # simple heuristic: if valf is large (e.g., 600000) and text contains 'lakh' near a smaller number like '6'
    if valf >= 1000:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lacs|lac)", text, flags=re.I)
        if m:
            try:
                base = float(m.group(1))
                conv = base * 100000
                if abs(conv - valf) / valf < 0.05:
                    return True, f"matched '{m.group(0)}' interpreted as {conv}"
            except Exception:
                pass

    return False, f"numeric value {valf} not found in text numbers: {text_nums}"


def token_overlap_reason(value: Any, text: str) -> Dict[str, Any]:
    vstr = "" if value is None else str(value)
    v_tokens = tokenize(vstr)
    t_tokens = tokenize(text)
    if not v_tokens:
        return {"reason": "empty_or_null_value", "matched_tokens": 0, "total_tokens": 0}
    set_v = set(v_tokens)
    set_t = set(t_tokens)
    common = set_v & set_t
    return {
        "reason": "token_overlap",
        "matched_tokens": len(common),
        "total_tokens": len(set_v),
        "matched_examples": list(common)[:10],
    }


def reason_for_exclusion(json_path: str, val: Any, input_text: str) -> Dict[str, Any]:
    """
    Return a structured explanation why the JSON leaf value was not found in the input text.
    """
    if val is None:
        return {"reason": "null_value_in_json", "detail": "JSON value is null"}
    if isinstance(val, str) and val.strip() == "":
        return {"reason": "empty_string_in_json", "detail": "JSON value is empty string"}
    # Lists/dicts: provide per-item check
    if isinstance(val, list):
        items = []
        for i, it in enumerate(val):
            found, snip = find_snippet(input_text, it)
            if found:
                items.append({"index": i, "value": it, "found": True, "snippet": snip})
            else:
                items.append({"index": i, "value": it, "found": False, "analysis": token_overlap_reason(it, input_text)})
        return {"reason": "list_values_mismatch", "detail": f"{len([x for x in items if x['found']])} of {len(items)} items found", "items": items}

    # Numeric heuristic
    num_ok, num_evidence = numeric_matches(val, input_text)
    if num_ok:
        return {"reason": "numeric_matched", "detail": num_evidence}
    # Exact / substring search
    found_exact, snip = find_snippet(input_text, val)
    if found_exact:
        return {"reason": "found_by_snippet", "detail": "substring matched", "snippet": snip}
    # Token overlap heuristic
    overlap = token_overlap_reason(val, input_text)
    if overlap["matched_tokens"] > 0:
        pct = overlap["matched_tokens"] / max(1, overlap["total_tokens"])
        return {"reason": "partial_token_match", "detail": f"{overlap['matched_tokens']}/{overlap['total_tokens']} tokens matched", "matched_examples": overlap["matched_examples"]}
    # final fallback
    return {"reason": "no_match", "detail": "no token, numeric, or substring match found"}


def split_paragraphs(text: str) -> List[str]:
    # split on 2+ newlines; keep paragraph order and trim
    parts = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in parts if p.strip()]
    return paragraphs


def analyze_paragraphs(paragraphs: List[str], flat_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Precompute token sets for JSON values for faster overlap checks
    json_tokens = {}
    for path, val in flat_json.items():
        json_tokens[path] = set(tokenize("" if val is None else str(val)))

    analyzed = []
    for idx, para in enumerate(paragraphs):
        p_tokens = set(tokenize(para))
        if not p_tokens:
            reason = {"reason": "empty_paragraph", "detail": "paragraph is empty after normalization"}
            analyzed.append({"index": idx, "paragraph": para, "matched": False, "reason": reason})
            continue

        # Find JSON paths with which paragraph shares tokens
        overlaps = []
        for path, jtoks in json_tokens.items():
            common = p_tokens & jtoks
            if common:
                overlaps.append({"json_path": path, "overlap_count": len(common), "common_tokens": list(common)[:10]})

        overlaps.sort(key=lambda x: -x["overlap_count"])

        if overlaps:
            # build detailed reason showing best matches and snippets
            best = overlaps[0]
            candidates = []
            for o in overlaps[:5]:
                # find snippet for matched json value in paragraph (if any)
                val = flat_json.get(o["json_path"])
                found, snip = find_snippet(para, val)
                candidates.append({
                    "json_path": o["json_path"],
                    "overlap_count": o["overlap_count"],
                    "common_tokens": o["common_tokens"],
                    "json_value": flat_json.get(o["json_path"]),
                    "snippet_in_paragraph": snip if found else None
                })
            reason = {
                "reason": "partial_overlap_with_json_values",
                "detail": f"{len(overlaps)} JSON leaf values share tokens with this paragraph",
                "top_candidates": candidates
            }
            analyzed.append({"index": idx, "paragraph": para, "matched": True, "reason": reason})
        else:
            # No token overlap with any JSON value — try numeric heuristic
            nums_in_para = extract_numbers_from_text(para)
            numeric_evidence = []
            if nums_in_para:
                for path, val in flat_json.items():
                    ok, ev = numeric_matches(val, para)
                    if ok:
                        numeric_evidence.append({"json_path": path, "evidence": ev})
            if numeric_evidence:
                reason = {"reason": "numeric_match_in_paragraph", "detail": numeric_evidence}
                analyzed.append({"index": idx, "paragraph": para, "matched": True, "reason": reason})
            else:
                # genuinely not represented
                reason = {
                    "reason": "not_represented_in_json",
                    "detail": "no token overlap, substring, or numeric match found between this paragraph and any JSON leaf values",
                    "example_tokens": list(p_tokens)[:20]
                }
                analyzed.append({"index": idx, "paragraph": para, "matched": False, "reason": reason})

    return analyzed


def build_comparison(input_text: str, output_json: Any) -> Dict[str, Any]:
    flat = flatten_json(output_json)
    included = []
    excluded = []
    for path, val in flat.items():
        found, snippet = find_snippet(input_text, val)
        entry = {"json_path": path, "value": val, "found_in_text": found}
        if found:
            entry["snippet"] = snippet
            included.append(entry)
        else:
            reason = reason_for_exclusion(path, val, input_text)
            entry["reason"] = reason
            excluded.append(entry)

    paragraphs = split_paragraphs(input_text)
    paragraph_analysis = analyze_paragraphs(paragraphs, flat)

    # detect proper nouns in text not in JSON values (for quick reference)
    pn_matches = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", input_text)
    pn_counts: Dict[str, int] = {}
    for p in pn_matches:
        pn_counts[p] = pn_counts.get(p, 0) + 1
    json_values_text = " ".join([str(v) for v in flat.values() if v is not None])
    not_represented = []
    for phrase, cnt in sorted(pn_counts.items(), key=lambda x: -x[1]):
        if phrase.lower() not in json_values_text.lower():
            not_represented.append({"phrase": phrase, "count": cnt})

    return {
        "summary": {
            "total_json_fields": len(flat),
            "included_fields": len(included),
            "excluded_fields": len(excluded),
            "total_paragraphs": len(paragraphs),
            "paragraphs_not_represented": len([p for p in paragraph_analysis if not p["matched"]])
        },
        "included": included,
        "excluded": excluded,
        "paragraph_analysis": paragraph_analysis,
        "text_only_proper_nouns": not_represented,
    }


def write_reports(report: Dict[str, Any], json_path: str, md_path: str) -> None:
    ensure_output_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    ensure_output_dir(md_path)
    with open(md_path, "w", encoding="utf-8") as f:
        s = report["summary"]
        f.write("# Comparison Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- total_json_fields: {s['total_json_fields']}\n")
        f.write(f"- included_fields: {s['included_fields']}\n")
        f.write(f"- excluded_fields: {s['excluded_fields']}\n")
        f.write(f"- total_paragraphs: {s['total_paragraphs']}\n")
        f.write(f"- paragraphs_not_represented: {s['paragraphs_not_represented']}\n\n")

        f.write("## Included JSON fields (found in text)\n\n")
        for it in report["included"]:
            f.write(f"- {it['json_path']}: {repr(it['value'])}\n")
            f.write(f"  - snippet: {it.get('snippet','')}\n")

        f.write("\n## Excluded JSON fields (NOT found in text) with reasons\n\n")
        for ex in report["excluded"]:
            f.write(f"- {ex['json_path']}: {repr(ex['value'])}\n")
            reason = ex.get("reason", {})
            f.write(f"  - reason: {reason.get('reason')}\n")
            if "detail" in reason:
                f.write(f"    - detail: {reason['detail']}\n")
            if "snippet" in reason:
                f.write(f"    - snippet: {reason['snippet']}\n")
            if "matched_examples" in reason:
                f.write(f"    - matched_examples: {reason['matched_examples']}\n")
            if "items" in reason:
                for it in reason["items"]:
                    if it["found"]:
                        f.write(f"    - item[{it['index']}] found: snippet: {it.get('snippet','')}\n")
                    else:
                        f.write(f"    - item[{it['index']}] NOT found: analysis: {it.get('analysis')}\n")

        f.write("\n## Paragraph-level analysis (paragraphs not represented in JSON with reasons)\n\n")
        for p in report["paragraph_analysis"]:
            if not p["matched"]:
                f.write(f"### Paragraph {p['index']}\n\n")
                f.write(p["paragraph"] + "\n\n")
                r = p["reason"]
                f.write(f"- reason: {r.get('reason')}\n")
                if "detail" in r:
                    if isinstance(r["detail"], (list, dict)):
                        f.write(f"  - detail: {json.dumps(r['detail'], ensure_ascii=False)}\n")
                    else:
                        f.write(f"  - detail: {r['detail']}\n")
                if "example_tokens" in r:
                    f.write(f"  - example_tokens: {r['example_tokens']}\n")
                f.write("\n")

        f.write("\n## Proper noun phrases present in text but not in JSON values\n\n")
        for p in report["text_only_proper_nouns"][:200]:
            f.write(f"- {p['phrase']} (count: {p['count']})\n")


def main():
    if not os.path.exists(INPUT_TEXT_PATH):
        print(f"Input text not found: {INPUT_TEXT_PATH}")
        return
    if not os.path.exists(OUTPUT_JSON_PATH):
        print(f"Output JSON not found: {OUTPUT_JSON_PATH}")
        return

    text = load_text(INPUT_TEXT_PATH)
    out_json = load_json(OUTPUT_JSON_PATH)

    report = build_comparison(text, out_json)
    write_reports(report, COMPARISON_JSON_PATH, COMPARISON_MD_PATH)

    print("Comparison reports written:")
    print(f"- JSON: {COMPARISON_JSON_PATH}")
    print(f"- MD:   {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()
