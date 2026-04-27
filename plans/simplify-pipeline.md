# Plan: Simplify aie323_pipeline.py

## Context

- **Repo**: `cat-breed-repo` (AIE323 course — Thai cat food packaging survey analysis)
- **File**: `aie323_pipeline.py` (616 lines)
- **Goal**: Remove over-engineering while preserving correct analysis output
- **Branch**: `simplify/pipeline` (created from `main`)
- **Exit strategy**: `git checkout main && git branch -D simplify/pipeline`

## Scope

**Keep (core pipeline):**
- Load raw CSV → rename columns by index → extract Target_Option
- Filter to cat owners only
- Likert scale → numeric encoding
- Ordinal encode Age, one-hot encode Gender/Marital
- ANOVA feature selection (remove RandomForest)
- 6 charts generation (sequential) — remove chart 7 (RF feature importance)
- Export cleaned CSV + summary

**Remove (over-engineering):**
- `PipelineStage` base class + OOP hierarchy
- Parallel data profiling (ThreadPoolExecutor 4 workers)
- Parallel ANOVA + RandomForest — keep ANOVA only
- Parallel chart generation — sequential loop
- `FigureCanvasAgg` thread-safety code
- `run_data_profiling()` with ThreadPoolExecutor
- Unused `rf_result` conditional in chart 7
- `diagnose()` method that is never called
- Excess config.json fields (load only what is needed)

## Steps

### Step 1 — Audit before changes
- Read `aie323_pipeline.py` and `config.json` in full
- Run `python aie323_pipeline.py` and verify all 7 charts exist + CSV exported
- Record output filenames

**Verification**: `ls *.png | wc -l` → 7, `ls cleaned_survey_data.csv` → exists

---

### Step 2 — Remove OOP class hierarchy
- Delete `PipelineStage` base class (lines 57-74)
- Delete `__init__`, `execute`, `diagnose` methods
- Keep the file flat — just functions

**Verification**: `python aie323_pipeline.py` still runs, same output

---

### Step 3 — Simplify data profiling
- Delete `run_data_profiling()` function (lines 80-132)
- Delete `run_data_profiling(df_raw)` call (line 164)
- Keep `df_raw.describe(include='all')` inline if any logging references it, otherwise remove profiling entirely

**Verification**: `python aie323_pipeline.py` still runs

---

### Step 4 — Remove parallel feature selection, keep ANOVA only
- Delete `run_random_forest_feature_selection()` function (lines 350-372)
- Rename `run_anova_feature_selection()` → `run_feature_selection()` and simplify (remove threading logic)
- Remove `ThreadPoolExecutor` block (lines 374-385)
- Store result as `anova_result`
- **Remove Chart 7 entirely** (`generate_chart_rf_importance`) — it depends on `rf_result` which is being deleted; RF was redundant over ANOVA anyway
- Update `top10_features` to derive from `anova_result`

**Verification**: `python aie323_pipeline.py` still runs, ANOVA results logged, 6 PNGs generated (chart7 gone)

---

### Step 5 — Sequential chart generation
- Delete `generate_chart_rf_importance()` function (lines 530-543) — depends on deleted `rf_result`
- Update `chart_generators` list: remove `generate_chart_rf_importance` (now 6 items)
- Replace ThreadPoolExecutor block (lines 556-563) with simple `for` loop:
  ```python
  for gen in chart_generators:
      gen()
  ```
- Remove `from matplotlib.backends.backend_agg import FigureCanvasAgg` import if only used for thread-safety
- Remove `as_completed` from `concurrent.futures` import if no longer used

**Verification**: `python aie323_pipeline.py` still runs, 6 PNGs generated

---

### Step 6 — Remove unused code and fix summary text
- Check which keys in `config.json` are actually used; remove unused ones or keep all if minimal impact
- Remove `diagnose()` method if any survive Step 2
- Update `presentation_summary.txt` to reflect changes:
  - "Feature Selection: ANOVA only (removed RandomForest)"
  - "Chart Generation: Sequential (removed parallel ThreadPoolExecutor)"
  - "Total charts generated: 6" (not 7)
- Ensure all imports are used

**Verification**: `python aie323_pipeline.py` still runs, same output

---

### Step 7 — Final verification
- Run `python aie323_pipeline.py` — confirm all 6 remaining charts + cleaned CSV produced
- Count lines: original ~616 → target ~250-300 (charts: 7→6)
- No `ThreadPoolExecutor` imports remaining
- No `PipelineStage` class remaining

**Verification**: All 6 remaining charts match Step 1 baseline, CSV schema unchanged, line count ~250-300

## Anti-Patterns to Avoid

- Do NOT add type annotations (not required for course project)
- Do NOT add docstrings beyond one-line for each function
- Do NOT refactor variable names (not part of simplification)
- Do NOT change chart output or CSV schema (must match original for presentation)
- Do NOT add tests (out of scope for course script)

## Rollback

If anything breaks at any step:
```bash
git checkout main && git branch -D simplify/pipeline
```
Start fresh with the plan.
