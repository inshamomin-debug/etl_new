import os
from dotenv import load_dotenv

# Load .env (optional)
load_dotenv()

# Paths used by run_pipeline.py — either set these in your .env or accept the defaults below
INPUT_TEXT_PATH = os.getenv("INPUT_TEXT_PATH", "data/input.txt")
SCHEMA_JSON_PATH = os.getenv("SCHEMA_JSON_PATH", "schema/template.json")
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", "output/result.json")