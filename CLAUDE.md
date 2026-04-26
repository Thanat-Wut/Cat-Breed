# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cat food packaging survey analysis pipeline for the AIE323 course. Determines the most preferred packaging design for a foreign cat food brand entering the Thai market.

## Commands

```bash
python aie323_pipeline.py
```

This runs the full pipeline: loads raw survey data, cleans/encodes it, performs feature selection via ANOVA, generates 6 charts, and exports `cleaned_survey_data.csv`.

## Architecture

### Main Pipeline (`aie323_pipeline.py`)

The pipeline processes survey data through 5 stages:

1. **Target Variable Identification** - Extracts first choice from `Top3_Choices` column (comma-separated)
2. **Data Cleaning** - Drops empty rows, filters to cat owners, standardizes breed/brand text
3. **Feature Selection & Encoding** - Maps Likert scales to numeric, ordinal-enodes age, one-hot encodes gender/marital; uses ANOVA to find top 10 features
4. **Data Visualization** - Generates 6 PNG charts
5. **Export** - Writes cleaned CSV and `presentation_summary.txt`

### Helper Scripts

- `analyze_data.py` - Explores raw column distributions and value frequencies
- `analyze_mapping.py` - Documents how corrupted Thai characters map back to original values (Thai text stored as `?` due to cp874 encoding)
- `check_breeds.py`, `check_encoding.py`, `check_utf8.py` - One-off verification scripts

### Data Flow

Raw CSV → `aie323_pipeline.py` → `cleaned_survey_data.csv` + `presentation_summary.txt` + 6 `chart*.png` files

### Key Encoding Issue

The survey CSV (cp874 encoding) has Thai characters corrupted to `?`. The pipeline renames columns by index and maps Likert values by string length rather than character content. See `analyze_mapping.py` for the complete character-to-length mapping reference.
