"""
config.py - Central configuration for the Telco Churn Analysis project.

All file paths are resolved relative to this file's location,
so the project works on any machine without editing individual scripts.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_PATH  = DATA_DIR / "raw" / "Dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "cleaned_dataset.csv"

# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR  = ROOT_DIR / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR   = ROOT_DIR / "models"

# ── SQL ───────────────────────────────────────────────────────────────────────
SQL_DIR = ROOT_DIR / "sql"

# ── Notebooks ─────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ── Ensure output directories exist on import ─────────────────────────────────
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
