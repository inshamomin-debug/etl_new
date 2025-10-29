# -*- coding: utf-8 -*-
import os
import json
import re
import sys
import subprocess
from pathlib import Path
from typing import List
from datetime import datetime
from dotenv import load_dotenv

# Optional import for Gemini client - don't fail import if package missing
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Try to import user's config module but tolerate it missing
try:
    import config
except Exception:
    config = None

# Load .env (optional)
load_dotenv()

def get_cfg(name: str, default):
    """Return value from config module if present, else from environment, else default."""
    if config and hasattr(config, name):
        return getattr(config, name)
    return os.getenv(name, default)

# Paths used by run_pipeline.py — either set these in your .env or accept the defaults below
INPUT_TEXT_PATH = get_cfg("INPUT_TEXT_PATH", "data/input.txt")
INPUT_DIR = get_cfg("INPUT_DIR", "data/inputs")
OUTPUT_DIR = get_cfg("OUTPUT_DIR", "output/results")
REPORTS_DIR = get_cfg("REPORTS_DIR", "output/reports")
SCHEMA_JSON_PATH = get_cfg("SCHEMA_JSON_PATH", "schema/template.json")
OUTPUT_JSON_PATH = get_cfg("OUTPUT_JSON_PATH", "output/result.json")

# --- 1. EXTRACT ---

def extract_data(text_path, schema_path):
    """
    Loads the raw text file and the JSON schema template.
    """
    print(f"Extracting data from {text_path} and {schema_path}...")
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = json.load(f)
            
        print("Data extraction successful.")
        return text_content, schema_content
        
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return None, None
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None, None

# --- 2. TRANSFORM ---

def _extract_special_tokens(text: str, max_tokens: int = 80):
    """Return a short list of tokens likely to be non-English / SMS / numeric artifacts to preserve verbatim."""
    if not text:
        return []
    # non-ASCII words (e.g., local language), tokens containing digits, and short sms-like tokens
    non_ascii = set(re.findall(r"\b[^\x00-\x7F]+\b", text))
    digit_tokens = set(re.findall(r"\b\S*\d+\S*\b", text))
    short_tokens = set([t for t in re.findall(r"\b\w{1,3}\b", text) if re.search(r"[a-zA-Z0-9]", t)])
    tokens = list(non_ascii | digit_tokens | short_tokens)
    # sort by occurrence order in text
    tokens_sorted = sorted(tokens, key=lambda t: text.find(t) if text.find(t) >= 0 else len(text))
    return tokens_sorted[:max_tokens]

def transform_text_to_json(text_content, schema_template):
    """
    Uses the Gemini API to populate the JSON schema with text content.
    """
    if not text_content or not schema_template:
        print("Transform step skipped due to missing data.")
        return None

    if genai is None:
        print("Error: google.generativeai (Gemini client) not installed or failed to import.")
        return None

    print("Starting transformation with Gemini API...")
    
    # Configure the API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment or .env file.")
        return None
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        return None

    # Set up the model
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"Error creating Gemini model: {e}")
        return None

   # build improved prompt that instructs preserving tokens and returning raw snippets
    prompt_intro = (
        "You are an expert agricultural data analyst. Your goal is to deeply and comprehensively populate the entire JSON schema using all relevant information from the INPUT TEXT.\n"
        "This is not just simple data extraction. You must analyze the text for context, advice, and explanations.\n\n"
        "Important instructions (must follow exactly):\n"
        " - Output MUST be valid JSON only (no explanation or markdown).\n"
        " - **Comprehensiveness is key:** Actively search for and populate qualitative data. This includes:\n"
        "   - **Solutions & Advice:** Fill all `tips`, `corrective`, `cure`, and `usage` fields with actionable advice, preventive measures, and specific solutions mentioned in the text (e.g., *how* to prevent a disease, *what* to do for heat stress, *how* to manage sapling shortages).\n"
        "   - **Explanations (The 'Why'):** Use `description` and `notes` fields to capture reasons and context (e.g., *why* a disease is a threat, *why* a practice is recommended, *why* fertigation is used in winter).\n"
        "   - **Strategic Goals:** Populate sections like `storage`, `transportation`, and `exporting` with any mentions of future goals or infrastructure (e.g., 'cold storage', 'international packaging standards').\n"
        " - **Field Matching:** Be meticulous in matching information to the detailed fields in the new schema. For example, the distinction between 'Cold Injury' and 'Fungal Blight' should be captured in their respective `description` or `symptoms` fields.\n"
        " - **Normalization:** If you normalize a numeric/unit value (like '9°C' or '5 kg'), include the original text under a '_raw' sibling.\n"
        " - **Preserve Tokens:** Preserve any token from the text that appears in the SPECIAL_TOKENS list exactly as-is (do not translate or normalize).\n"
        " - **Completeness:** Do not invent facts. Do not drop any unique provided field unless it is an exact duplicate.\n"
        " - **Multiple Values:** When multiple values exist for the same logical field (like `symptoms` or `tips`), include all distinct values (as a list) and preserve the order of appearance.\n"
        "\n"
    )

    # --- ADDED: compute special_tokens from the input text to avoid NameError ---
    special_tokens = _extract_special_tokens(text_content)
    # optional: limit or log
    # print(f"Special tokens extracted: {len(special_tokens)} tokens")

    prompt = (
        prompt_intro
        + "SPECIAL_TOKENS (preserve verbatim):\n"
        + json.dumps(special_tokens, ensure_ascii=False, indent=2)
        + "\n\n--- JSON SCHEMA TEMPLATE ---\n"
        + json.dumps(schema_template, indent=2, ensure_ascii=False)
        + "\n\n--- INPUT TEXT ---\n"
        + text_content
        + "\n\n--- POPULATED JSON OUTPUT ---\n"
    )

    try:
        # deterministic config for stable extraction
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.0)
        response = model.generate_content(prompt, generation_config=generation_config)
        print("Transformation successful. Received JSON from API.")
        return response.text
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# --- 3. LOAD ---

def _parse_number(s: str):
    """Try to parse a number from string (handles commas and dots)."""
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def _normalize_string_unit(s: str):
    """Return a normalized representation for common unit patterns found in text.
    For ranges returns {'min':..,'max':..,'unit':..}, for single values returns {'value':..,'unit':..},
    for plain numbers returns number, otherwise None.
    """
    if not isinstance(s, str):
        return None

    s_strip = s.strip()
    # handle lakhs/lacs
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:lakh|lac|lacs|lakhs)\b", s_strip, flags=re.I)
    if m:
        base = _parse_number(m.group(1))
        if base is not None:
            return {"value": int(base * 100000), "unit": "count_lakh"}  # context dependent

    # currency INR (e.g., "₹ 3,300" or "Rs. 3300")
    m = re.search(r"(?:₹|Rs\.?|INR)\s*([0-9\.,]+)", s_strip)
    if m:
        v = _parse_number(m.group(1))
        if v is not None:
            return {"value": v, "unit": "INR"}

    # temperature range Celsius (e.g., "25-26 C")
    m = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*-\s*(-?\d+(?:[\.,]\d+)?)\s*(?:°|\s)?\s*C\b", s_strip, flags=re.I)
    if m:
        a = _parse_number(m.group(1))
        b = _parse_number(m.group(2))
        if a is not None and b is not None:
            return {"min": a, "max": b, "unit": "C"}

    # temperature Celsius (single)
    m = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*(?:°|\s)?\s*C\b", s_strip, flags=re.I)
    if m:
        v = _parse_number(m.group(1))
        if v is not None:
            return {"value": v, "unit": "C"}

    # liters (single or range)
    m = re.match(r"^(\d+(?:[\.,]\d+)?)(?:\s*-\s*(\d+(?:[\.,]\d+)?))?\s*(?:liters|litre|liter|l)\b", s_strip, flags=re.I)
    if m:
        a = _parse_number(m.group(1))
        b = _parse_number(m.group(2)) if m.group(2) else None
        if a is not None and b is not None:
            return {"min": a, "max": b, "unit": "L"}
        if a is not None:
            return {"value": a, "unit": "L"}
            
    # kg (single or range)
    m = re.match(r"^(\d+(?:[\.,]\d+)?)(?:\s*-\s*(\d+(?:[\.,]\d+)?))?\s*(?:kg|kilograms)\b", s_strip, flags=re.I)
    if m:
        a = _parse_number(m.group(1))
        b = _parse_number(m.group(2)) if m.group(2) else None
        if a is not None and b is not None:
            return {"min": a, "max": b, "unit": "kg"}
        if a is not None:
            return {"value": a, "unit": "kg"}

    # tons / tonnes
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:tons|tonnes|t)\b", s_strip, flags=re.I)
    if m:
        v = _parse_number(m.group(1))
        if v is not None:
            return {"value": v, "unit": "tons"}

    # plain numeric with possible % or unitless
    m = re.match(r"^([0-9\.,]+)$", s_strip)
    if m:
        v = _parse_number(m.group(1))
        if v is not None:
            return v

    # ranges like "18-20" without explicit unit
    m = re.match(r"^(\d+(?:[\.,]\d+)?)\s*-\s*(\d+(?:[\.,]\d+)?)$", s_strip)
    if m:
        a = _parse_number(m.group(1)); b = _parse_number(m.group(2))
        if a is not None and b is not None:
            return {"min": a, "max": b}

    return None

def normalize_units_in_obj(obj, preserve_raw=False):
    """Recursively normalize strings that contain units into structured dicts."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = normalize_units_in_obj(v, preserve_raw)
        return obj
    if isinstance(obj, list):
        return [normalize_units_in_obj(i, preserve_raw) for i in obj]
    if isinstance(obj, str):
        norm = _normalize_string_unit(obj)
        if norm is not None:
            # norm can be dict, number or other scalar — handle safely
            if isinstance(norm, dict):
                # merge normalized dict and optionally keep raw string
                if preserve_raw:
                    merged = dict(norm)
                    merged["_raw"] = obj
                    return merged
                return dict(norm)
            else:
                # scalar (int/float/str) normalized value
                if preserve_raw:
                    return {"_raw": obj, "value": norm}
                return norm
        # try to coerce plain numbers in string
        num = _parse_number(obj)
        if num is not None:
            return num
        return obj
    return obj

def load_json_output(json_string, output_path):
    """
    Saves the transformed JSON string to the output file.
    Normalizes units (liters, Celsius (C), INR, lakhs, tons) into structured values.
    """
    if not json_string:
        print("Load step skipped due to missing transformed data.")
        return

    print(f"Loading data into {output_path}...")
    try:
        # Parse the JSON string from the AI to validate it
        data = json.loads(json_string)

        # Normalize units throughout the parsed object
        data = normalize_units_in_obj(data, preserve_raw=True)

        # Write the formatted JSON to the output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Data load complete. Output file created.")
        
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from AI response. Output was:")
        print(json_string)
    except Exception as e:
        print(f"Error during load: {e}")

# --- Main Pipeline Runner ---

def select_input_files(input_dir: str) -> List[Path]:
    """
    Interactive selection: list .txt files in input_dir and let the user choose.
    Returns list of Path objects (empty list = nothing selected / quit).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory not found: {input_path}")
        return []

    files = sorted(input_path.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_path}.")
        return []

    while True:
        print("\nAvailable input files:")
        for i, f in enumerate(files, start=1):
            print(f"  {i}) {f.name}")
        print("  a) All files")
        print("  q) Quit / cancel")

        choice = input("Select file numbers (e.g. 1,3-5), 'a' for all, or 'q' to quit: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return []
        if choice in ("a", "all"):
            return files

        # parse numbers / ranges
        selected_indices = set()
        valid = True
        for part in re.split(r"\s*,\s*", choice):
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    a_i = int(a); b_i = int(b)
                    if a_i < 1 or b_i > len(files) or a_i > b_i:
                        valid = False; break
                    for idx in range(a_i, b_i + 1):
                        selected_indices.add(idx)
                except Exception:
                    valid = False; break
            else:
                try:
                    n = int(part)
                    if n < 1 or n > len(files):
                        valid = False; break
                    selected_indices.add(n)
                except Exception:
                    valid = False; break

        if not valid or not selected_indices:
            print("Invalid selection. Please try again.")
            continue

        selected = [files[i - 1] for i in sorted(selected_indices)]
        print("You selected:")
        for f in selected:
            print(f"  - {f.name}")
        confirm = input("Proceed with these files? (y/n): ").strip().lower()
        if confirm in ("y", "yes"):
            return selected
        # otherwise loop again

def run_etl_pipeline():
    """Runs the ETL pipeline for user-selected input files."""
    print("--- Starting interactive Text-to-JSON ETL Pipeline ---")

    INPUT_DIR = getattr(__import__("config"), "INPUT_DIR", os.getenv("INPUT_DIR", "data/inputs"))
    OUTPUT_DIR = getattr(__import__("config"), "OUTPUT_DIR", os.getenv("OUTPUT_DIR", "output/results"))
    REPORTS_DIR = getattr(__import__("config"), "REPORTS_DIR", os.getenv("REPORTS_DIR", "output/reports"))
    SCHEMA_JSON_PATH = getattr(__import__("config"), "SCHEMA_JSON_PATH", os.getenv("SCHEMA_JSON_PATH", "schema/template.json"))

    selected_files = select_input_files(INPUT_DIR)
    if not selected_files:
        print("No files selected. Exiting.")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    for txt_file in selected_files:
        print(f"\n--- Processing: {txt_file.name} ---")
        text, schema = extract_data(str(txt_file), SCHEMA_JSON_PATH)
        if not text or schema is None:
            print(f"Skipping {txt_file.name} due to extraction error.")
            continue

        json_output_string = transform_text_to_json(text, schema)
        if json_output_string is None:
            print(f"Skipping {txt_file.name} due to transform error.")
            continue

        # create timestamped output filenames
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json_path = Path(OUTPUT_DIR) / f"{txt_file.stem}_{ts}.json"
        load_json_output(json_output_string, str(out_json_path))

        # Run comparator for this pair (input, output) and write per-file reports
        try:
            comparison_json = Path(REPORTS_DIR) / f"{txt_file.stem}_comparison_{ts}.json"
            comparison_md = Path(REPORTS_DIR) / f"{txt_file.stem}_comparison_{ts}.md"

            env = os.environ.copy()
            env["INPUT_TEXT_PATH"] = str(txt_file)
            env["OUTPUT_JSON_PATH"] = str(out_json_path)
            env["COMPARISON_JSON_PATH"] = str(comparison_json)
            env["COMPARISON_MD_PATH"] = str(comparison_md)

            subprocess.run([sys.executable, str(Path(__file__).with_name("compare_report.py"))], check=False, env=env)
            print(f"Report generated for {txt_file.name} -> {comparison_md}")
        except Exception as e:
            print(f"Warning: failed to run comparator for {txt_file.name}: {e}")

    print("\n--- ETL Pipeline Finished ---")
    
if __name__ == "__main__":
    run_etl_pipeline()