# Text-to-JSON ETL Pipeline (Full Documentation)

This repository turns unstructured agricultural text into clean, normalized JSON according to a schema, then explains exactly what was captured and what was missed with a detailed comparison report. It supports single-file and batch processing, unit normalization, result merging, and auditability through optional provenance fields.

## Key capabilities
- **Schema-driven extraction**: Uses `schema/template.json` to guide model output.
- **Normalization**: Converts noisy text values into consistent numbers, ranges, and units.
- **Traceability**: Optional `source_snippet` and `confidence` per field.
- **Explainability**: Machine- and human-readable comparison reports.
- **Batch-ready**: Process many `.txt` files from `data/inputs/`.
- **Merging**: Combine compatible outputs into a single JSON.

## Repository layout
- `data/`
  - `inputs/` — raw `.txt` files for batch processing (examples included)
- `schema/`
  - `template.json` — schema template used to shape extraction
- `output/`
  - `results/` — per-input JSON results with timestamps
  - `merged/` — merged JSON files with encoded provenance in the filename
  - `reports/` — comparison reports (`.json` and `.md`)
  - `result.json` — convenience latest result for single runs
- `run_pipeline.py` — main ETL (Extract → Transform → Load)
- `compare_report.py` — compares source text vs JSON and explains gaps
- `merge_json.py` — merges multiple result files
- `config.py` — optional centralized configuration
- `requirements.txt` — dependencies

## Supported environments
- Python 3.8+
- Windows 10/11 (PowerShell examples provided)
- Works on Linux/macOS with equivalent shell commands

## Quick start (Windows PowerShell)
1) Open the project folder
```
cd "C:\Users\Lenovo\Desktop\ETL_NEW"
```
2) Create and activate a virtual environment (recommended)
```
python -m venv venv
./venv/Scripts/Activate
```
3) Install dependencies
```
pip install -r requirements.txt
```

## Configuration
Set via environment variables or a `.env` file in the project root (environment has precedence). You may also hardcode defaults in `config.py`.

Common variables:
- `INPUT_TEXT_PATH` — path to a single input file (default may be `data/input.txt`). For batch runs, place files in `data/inputs/`.
- `SCHEMA_JSON_PATH` — path to `schema/template.json`.
- `OUTPUT_JSON_PATH` — single-run output path (default `output/result.json`).
- `GEMINI_API_KEY` — required if using Gemini via `google-generativeai`.
- `COMPARISON_JSON_PATH` — optional override for JSON report output.
- `COMPARISON_MD_PATH` — optional override for Markdown report output.

Example `.env`:
```
INPUT_TEXT_PATH=data/input.txt
SCHEMA_JSON_PATH=schema/template.json
OUTPUT_JSON_PATH=output/result.json
GEMINI_API_KEY=sk-...
```

## Architecture and data flow
1. Extract
   - Read raw source text from `INPUT_TEXT_PATH` or from all `.txt` files in `data/inputs/` (batch).
   - Load `schema/template.json`.
2. Transform
   - Invoke LLM with a strict JSON-only prompt to fill the schema.
   - Parse/validate returned JSON, apply fallbacks if needed.
3. Normalize
   - Convert units, numeric strings, and ranges into consistent structures.
   - Optionally preserve the original value as `_raw` for auditing.
4. Load
   - Write result(s) into `output/result.json` and/or `output/results/<name>_<timestamp>.json`.
   - Optionally run comparison and merging steps.

## Running the pipeline
Single file (from `.env` paths):
```
python .\run_pipeline.py
```
Batch mode (typical):
- Place multiple `.txt` files in `data/inputs/`.
- Run the same command; outputs go to `output/results/` with timestamped filenames.

Outputs:
- Latest/single: `output/result.json`
- Per-input: `output/results/<inputBase>_<YYYYMMDD_HHMMSS>.json`

## Generating comparison reports
Run after results exist:
```
python .\compare_report.py
```
Produces:
- JSON report: `output/reports/<name>_comparison_<timestamp>.json`
- Markdown report: `output/reports/<name>_comparison_<timestamp>.md`

Report contents:
- `summary`: counts for fields, included/excluded, paragraph coverage.
- `included`: JSON leaf values confirmed in source with snippet alignment.
- `excluded`: values not matched with structured reasons (null, mismatch, partial overlap, no match).
- `paragraph_analysis`: coverage of each paragraph and notable omissions.
- `text_only_proper_nouns`: proper nouns in text absent from JSON.

## Merging results
Combine compatible outputs:
```
python .\merge_json.py
```
Merged files land in `output/merged/` with a filename that lists merged sources and timestamps.

## Prompting and schema examples
Schema fragment (example):
```json
{
  "crop": {
    "name": "",
    "varieties": [],
    "planting": {
      "spacing": { "row_cm": null, "plant_cm": null },
      "season": ""
    },
    "irrigation": {
      "liters_per_plant": null,
      "instructions": ""
    },
    "treatments": {
      "pesticides": [],
      "timings": []
    }
  }
}
```

Prompt guidelines:
- Instruct the model to return JSON only, matching the schema shape strictly.
- Omit unknowns or set to null/empty arrays/strings as appropriate.
- Optionally include `source_snippet` and `confidence` per field for traceability.

## Normalization rules (details)
Implemented in `run_pipeline.py` (function `_normalize_string_unit` and callers):
- Numeric strings → numbers when unambiguous.
- Ranges like `18-20` → `{ "min": 18, "max": 20 }`.
- Volume: `5 L`, `5 liters` → unit `"L"` with value (or min/max for ranges).
- Temperature: `25°C` → unit `"C"`.
- Currency: `₹500`, `Rs 500`, `INR 500` → unit `"INR"`, numeric value only.
- Lakhs: `6 lakh` → value × 100000, optionally annotate `_raw`.

Extending normalization:
- Add parsing branches in `_normalize_string_unit` for new units (e.g., acres, kg, m³).
- Decide representation: scalar + `unit`, or nested object with `value`, `unit`, and `{min,max}`.

## Comparison logic (high level)
- Token-based and numeric heuristics match JSON leaf values to text snippets.
- Reasons for exclusions are categorized to aid debugging and schema refinement.
- Sensitivity can be tuned by adjusting tokenization and numeric match thresholds in `compare_report.py`.

## Performance tips
- Keep prompts minimal and schema-focused.
- Batch inputs to reduce overhead if your model/client supports it.
- Cache model responses during debugging to avoid repeated calls.

## Testing and reproducibility
- Pin dependency versions in `requirements.txt` as needed.
- Save raw model responses to `output/last_model_response.txt` during debugging.
- Use consistent input files and schema to compare runs.

## Security and privacy
- Do not commit real API keys. Use `.env` or secure secret stores.
- Review input texts for sensitive data before sending to external LLMs.
- If required, implement redaction before calling the model.

## Troubleshooting
- File not found
  - Verify `data/inputs/` and `schema/template.json`, or override paths in `.env`/`config.py`.
- JSON decode error from model
  - Enforce JSON-only outputs. Log raw responses for inspection and refine prompts.
- Missing/invalid `GEMINI_API_KEY`
  - Set environment variable and validate your client configuration.
- "argument of type 'float' is not iterable"
  - Ensure normalization handles scalar vs dict distinctly.
- Too many exclusions in reports
  - Include `source_snippet` in model outputs, increase fuzzy tolerance, or expand schema fields.

## FAQ
- Can I process many files automatically?
  - Yes. Place `.txt` files in `data/inputs/` and run the pipeline.
- How do I add a new unit type?
  - Extend `_normalize_string_unit` in `run_pipeline.py` and update downstream handling if needed.
- Can I merge from different sources?
  - Yes. Use `merge_json.py` and resolve conflicts manually or extend the strategy.
- Where are reports saved?
  - `output/reports/` with timestamped names for each source.

## Example workflow
1) Add inputs to `data/inputs/` (e.g., `Banana planting to harvesting planning.txt`).
2) Run pipeline:
```
python .\run_pipeline.py
```
3) Inspect `output/results/<name>_<timestamp>.json` (and `output/result.json`).
4) Create reports:
```
python .\compare_report.py
```
5) Merge multiple outputs:
```
python .\merge_json.py
```

## Contributing
- Keep code readable and documented.
- When adding normalization rules, include a brief rationale and tests/examples.
- Open issues with example input text and desired schema fields for guidance.


#
