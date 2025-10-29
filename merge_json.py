import os
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set

# make dotenv optional
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

# try to import Gemini client (optional)
try:
    import google.generativeai as genai
except Exception:
    genai = None

JSON_DIR = os.getenv("JSON_DIR", "output/results")
OUT_DIR = os.getenv("OUT_DIR", "output/merged")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def list_json_files(dirpath: str) -> List[Path]:
    p = Path(dirpath)
    if not p.exists():
        return []
    return sorted(p.glob("*.json"))


def select_files(files: List[Path]) -> List[Path]:
    if not files:
        print(f"No JSON files found in {JSON_DIR}")
        return []
    print("Available JSON files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {f.name}")
    print("  a) All")
    print("  q) Quit")
    while True:
        choice = input("Select files (e.g. 1,3-5), 'a' for all, or 'q' to quit: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return []
        if choice in ("a", "all"):
            return files
        parts = re.split(r"\s*,\s*", choice)
        idxs: Set[int] = set()
        ok = True
        for part in parts:
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = map(int, part.split("-", 1))
                    if a < 1 or b > len(files) or a > b:
                        ok = False
                        break
                    idxs.update(range(a, b + 1))
                except Exception:
                    ok = False
                    break
            else:
                try:
                    n = int(part)
                    if n < 1 or n > len(files):
                        ok = False
                        break
                    idxs.add(n)
                except Exception:
                    ok = False
                    break
        if not ok or not idxs:
            print("Invalid selection, try again.")
            continue
        selected = [files[i - 1] for i in sorted(idxs)]
        print("Selected:")
        for s in selected:
            print("  -", s.name)
        if input("Proceed? (y/n): ").strip().lower() in ("y", "yes"):
            return selected


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe_list_preserve_order(items: List[Any]) -> List[Any]:
    out = []
    seen = set()
    for it in items:
        # use canonical key for robust duplicate detection
        key = _canonical_key(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _local_merge_preserve_all(a: Any, b: Any) -> Any:
    """Local deterministic merge that preserves everything but deduplicates repeated items when possible."""
    # both dicts -> recursive merge
    if isinstance(a, dict) and isinstance(b, dict):
        out: Dict[str, Any] = {}
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            if k in a and k in b:
                out[k] = _local_merge_preserve_all(a[k], b[k])
            elif k in a:
                out[k] = a[k]
            else:
                out[k] = b[k]
        return out

    # both lists -> concat then dedupe via canonical keys
    if isinstance(a, list) and isinstance(b, list):
        return _dedupe_list_preserve_order(a + b)

    # one list, one scalar/dict -> append and dedupe
    if isinstance(a, list) and not isinstance(b, list):
        return _dedupe_list_preserve_order(a + [b])
    if isinstance(b, list) and not isinstance(a, list):
        return _dedupe_list_preserve_order([a] + b)

    # scalars or mixed types -> if equal return scalar, else return list of unique values (preserve order)
    if a == b:
        return a
    combined = []
    seen_keys = set()
    for v in (a, b):
        key = _canonical_key(v)
        if key not in seen_keys:
            seen_keys.add(key)
            combined.append(v)
    return combined


# ----- ADDED: canonical key helper and improved recursive dedupe -----
def _canonical_key(obj: Any) -> str:
    """
    Return a stable string key for obj used for deduplication.
    - dicts/lists -> JSON with sorted keys
    - strings -> stripped (preserve case)
    - numbers/bool/null -> JSON
    """
    if isinstance(obj, dict):
        # canonicalize child values too
        canon = {k: json.loads(_canonical_key(obj[k])) if isinstance(_canonical_key(obj[k]), str) and _looks_like_json(_canonical_key(obj[k])) else obj[k] for k in sorted(obj.keys())}
        return json.dumps(canon, sort_keys=True, ensure_ascii=False)
    if isinstance(obj, list):
        return json.dumps([_canonical_key(i) for i in obj], ensure_ascii=False)
    if isinstance(obj, str):
        return obj.strip()
    # numbers / booleans / None
    return json.dumps(obj, ensure_ascii=False)


def _looks_like_json(s: str) -> bool:
    """Helper to check if a string is JSON-ish (used internally)."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _dedupe_recursive(obj: Any) -> Any:
    """
    Recursively walk the merged structure and:
      - deduplicate items in lists (preserve order) using canonical keys
      - recurse into dict/list children
    """
    if isinstance(obj, dict):
        return {k: _dedupe_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        seen = set()
        out = []
        for item in obj:
            norm = _dedupe_recursive(item)
            key = _canonical_key(norm)
            if key not in seen:
                seen.add(key)
                out.append(norm)
        return out
    return obj


def call_gemini_merge(objects: List[Any]) -> Any:
    """Call Gemini to merge list of JSON objects into a single deduplicated JSON.
    Returns parsed JSON object or raises Exception.
    """
    if genai is None:
        raise RuntimeError("Gemini client not available")

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to configure Gemini client: {e}")

    # Build a stricter prompt that enforces canonical deduplication rules and gives short examples.
    prompt = (
        "You are a data engineer. Merge the provided JSON documents into ONE JSON object.\n\n"
        "STRICT RULES (must follow exactly):\n"
        "1) Remove exact and near-duplicate information so each unique fact appears only ONCE.\n"
        "   - For strings: compare after trimming whitespace and normalizing internal whitespace.\n"
        "     Consider strings equal if they match case-insensitively after trimming. When duplicates\n"
        "     are removed, KEEP the first-occurrence original form (preserve original casing/format of first seen).\n"
        "   - For numbers: treat numerically-equal values as duplicates (e.g. 1 and 1.0).\n"
        "   - For objects/lists: consider deep structural equality (keys/values) ignoring key order.\n"
        "2) For lists: produce a single list with unique items only, preserving the order of first appearance.\n"
        "   - If list items are objects, treat items as equal when their canonical structure and values match.\n"
        "3) For the same key appearing in multiple inputs:\n"
        "   - If all values are identical (by canonical rules above), keep a single value.\n"
        "   - If values differ, return a list of unique values (order = first occurrence across inputs).\n"
        "   - If values are objects, recursively merge them according to these rules.\n"
        "4) Do NOT invent, normalize meaningfully, or remove distinct fields. Only remove duplicates.\n"
        "5) Output MUST be valid JSON only (no explanation, no extra text). Use compact, valid JSON.\n\n"
        "SHORT EXAMPLES (illustrative):\n"
        "INPUT: [{\"name\":\"Banana\"},{\"name\":\" banana \"}] -> OUTPUT: {\"name\":\"Banana\"}\n"
        "INPUT: [{\"qty\":1},{\"qty\":1.0}] -> OUTPUT: {\"qty\":1}\n"
        "INPUT: [{\"tags\":[\"a\",\"b\"]},{\"tags\":[\"b\",\"c\"]}] -> OUTPUT: {\"tags\":[\"a\",\"b\",\"c\"]}\n"
        "INPUT: [{\"price\":\"₹ 1000\"},{\"price\":\"1000\"}] -> treat numeric/currency duplicates if clearly same number (prefer original first format).\n\n"
        "Now merge the following input JSON documents following these rules precisely.\n\n"
        "Input JSON documents:\n"
    )

    prompt += json.dumps(objects, ensure_ascii=False, indent=2)
    prompt += "\n\n---\nOutput the merged JSON now (only JSON):"

    try:
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        merged_text = response.text
        return json.loads(merged_text)
    except Exception as e:
        raise RuntimeError(f"Gemini merge failed: {e}")


def merge_selected_files_with_gemini(selected: List[Path]) -> Any:
    # load all selected JSON objects into list
    objects = []
    for p in selected:
        try:
            data = load_json(p)
        except Exception as e:
            print(f"Warning: failed to load {p.name}: {e} — skipping")
            continue
        objects.append(data)

    if not objects:
        return None

    # try Gemini first
    try:
        print("Calling Gemini to merge and deduplicate JSON files...")
        merged = call_gemini_merge(objects)
        print("Gemini merge successful.")
        # ensure merged result has list-level duplicates removed
        merged = _dedupe_recursive(merged)
        return merged
    except Exception as e:
        print(f"Gemini merge failed or unavailable: {e}")
        print("Falling back to local deterministic merge (preserve all, dedupe lists).")
        # fallback: merge pairwise using local deterministic merge
        merged = objects[0]
        for obj in objects[1:]:
            merged = _local_merge_preserve_all(merged, obj)
        # run recursive dedupe to ensure repeating information appears once
        merged = _dedupe_recursive(merged)
        return merged


def save_merged(merged: Any, out_dir: str, selected: List[Path]) -> Path:
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stems = "_".join([p.stem for p in selected])
    filename = f"merged_{stems}_{ts}.json"
    out_path = out_path_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    files = list_json_files(JSON_DIR)
    selected = select_files(files)
    if not selected:
        print("No files chosen. Exiting.")
        return
    merged = merge_selected_files_with_gemini(selected)
    if merged is None:
        print("No data merged. Exiting.")
        return
    out_path = save_merged(merged, OUT_DIR, selected)
    print(f"Merged JSON written to: {out_path}")


if __name__ == "__main__":
    main()