"""
AIE323 - Data Preparation Pipeline
===================================
โปรเจกต์วิเคราะห์ข้อมูลแบบสอบถาม (Survey Data)
เรื่องการออกแบบบรรจุภัณฑ์อาหารแมวสำหรับแบรนด์ต่างประเทศที่ต้องการบุกตลาดไทย

Hybrid Chain & Parallel Execution:
- Phase 1: Data Profiling (parallel)
- Phase 2: Pipeline Stages (chain with internal parallel)
- Phase 3: Live Diagnostics (after each stage)
"""

import pandas as pd
import numpy as np

# Set matplotlib backend to Agg BEFORE importing pyplot (fixes tkinter thread issue on Windows)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import resample
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import logging
import sys
import time
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG: Logging Setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================================================
# CONFIG: ตั้งค่า Font ภาษาไทย (ปรับ path ตามเครื่อง)
# ============================================================
try:
    plt.rcParams['font.family'] = 'Tahoma'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# PIPELINE STAGE BASE CLASS
# ============================================================
class PipelineStage:
    """Base class for pipeline stages with chain dependencies"""
    name: str = "BaseStage"

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def execute(self, context: dict) -> dict:
        """Execute the stage. Override in subclass."""
        raise NotImplementedError

    def diagnose(self, context: dict, result: dict) -> str:
        """Return diagnostic string after execution"""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        return f"{self.name}: completed in {duration:.2f}s"


def _safe_classification_report(y_true, y_pred, labels=None, target_names=None):
    """Return a dict report with zero_division protection."""
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def _flatten_classification_report(report_dict, target_name, model_name, step_name, threshold=None):
    """Flatten sklearn classification_report output into CSV-friendly rows."""
    rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append({
                'Target': target_name,
                'Model': model_name,
                'Step': step_name,
                'Label': label,
                'Threshold': threshold,
                'Precision': metrics.get('precision'),
                'Recall': metrics.get('recall'),
                'F1': metrics.get('f1-score'),
                'Support': metrics.get('support'),
            })
    return rows


def _random_oversample(X_train, y_train, random_state=42):
    """Simple train-fold oversampling fallback when imblearn is unavailable."""
    X_train = pd.DataFrame(X_train).copy()
    y_train = pd.Series(y_train).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    counts = y_train.value_counts()
    if len(counts) < 2:
        return X_train, y_train
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_to_add = counts.max() - counts.min()
    if n_to_add <= 0:
        return X_train, y_train

    minority_idx = y_train[y_train == minority_class].index
    sampled_idx = resample(
        minority_idx,
        replace=True,
        n_samples=n_to_add,
        random_state=random_state,
    )
    X_extra = X_train.loc[sampled_idx].reset_index(drop=True)
    y_extra = y_train.loc[sampled_idx].reset_index(drop=True)
    X_bal = pd.concat([X_train, X_extra], ignore_index=True)
    y_bal = pd.concat([y_train, y_extra], ignore_index=True)
    return X_bal, y_bal


def _build_oof_predictions(estimator, X, y, cv, threshold=0.5, positive_label=1, oversample=False):
    """Repeated CV evaluator returning pooled OOF predictions and fold-level importances."""
    y_series = pd.Series(y).reset_index(drop=True)
    X_df = pd.DataFrame(X).reset_index(drop=True)
    all_true = []
    all_pred = []
    all_score = []
    fold_importances = []
    fold_metrics = []
    fold_sizes = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y_series), start=1):
        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        y_train = y_series.iloc[train_idx].reset_index(drop=True)
        X_test = X_df.iloc[test_idx].reset_index(drop=True)
        y_test = y_series.iloc[test_idx].reset_index(drop=True)

        if oversample:
            X_train, y_train = _random_oversample(X_train, y_train, random_state=42 + fold_idx)

        model = clone(estimator)
        model.fit(X_train, y_train)

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                classes = list(model.classes_)
                if positive_label in classes:
                    pos_idx = classes.index(positive_label)
                else:
                    pos_idx = 1
                score = proba[:, pos_idx]
            else:
                score = np.asarray(proba).reshape(-1)
        else:
            score = None

        if score is None:
            pred = model.predict(X_test)
            score = pred.astype(float)
        else:
            pred = (score >= threshold).astype(int) if len(np.unique(y_series)) == 2 else model.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        all_score.extend(score.tolist())
        fold_sizes.append(len(test_idx))

        if hasattr(model, 'feature_importances_'):
            fold_importances.append(model.feature_importances_)

        if len(np.unique(y_series)) == 2:
            fold_metrics.append({
                'fold': fold_idx,
                'f1': f1_score(y_test, pred, zero_division=0),
                'balanced_acc': balanced_accuracy_score(y_test, pred),
                'accuracy': accuracy_score(y_test, pred),
            })
        else:
            fold_metrics.append({
                'fold': fold_idx,
                'macro_f1': f1_score(y_test, pred, average='macro', zero_division=0),
                'balanced_acc': balanced_accuracy_score(y_test, pred),
                'accuracy': accuracy_score(y_test, pred),
            })

    pooled = {
        'y_true': np.asarray(all_true),
        'y_pred': np.asarray(all_pred),
        'y_score': np.asarray(all_score),
        'fold_metrics': pd.DataFrame(fold_metrics),
        'fold_importances': fold_importances,
        'fold_sizes': fold_sizes,
    }
    return pooled


def _best_threshold_from_pr(y_true, y_score):
    """Pick the PR threshold that maximizes F1 on pooled out-of-fold predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, {'precision': precision, 'recall': recall, 'thresholds': thresholds}
    f1_scores = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'best_idx': best_idx,
    }


def _build_feature_frame(df_in, target_option_num=None):
    """Create model feature matrix for either multiclass preference or a binary intent target."""
    numeric_cols = [
        'Factor_Taste', 'Factor_Brand_Rep', 'Factor_Price', 'Factor_Ingredients', 'Factor_Packaging',
        'PkgFactor_Taste', 'PkgFactor_Freshness', 'PkgFactor_Nutrition', 'PkgFactor_Safety',
        'PkgFactor_Convenience', 'PkgFactor_Promotion', 'PkgFactor_Quality', 'PkgFactor_Price',
        'Age_Ordinal'
    ]
    numeric_cols += [c for c in df_in.columns if c.startswith('Gender_') and c not in {'Gender_Label'}]
    numeric_cols += [c for c in df_in.columns if c.startswith('Marital_') and c not in {'Marital_Label'}]
    numeric_cols += [c for c in df_in.columns if c in {'Need_Big_Text', 'Need_Interactive', 'Need_Clear_Window', 'Need_Pour_Lid', 'Need_Sodium_Info', 'Design_Special', 'Need_Functional_Pkg'}]

    if target_option_num is None:
        option_feature_cols = [
            c for c in df_in.columns
            if c.startswith('Opt')
            and c.endswith(('Attractiveness', 'Trust', 'Modernity', 'Premium_Feel'))
        ]
    else:
        option_feature_cols = [
            f'Opt{target_option_num}_Attractiveness',
            f'Opt{target_option_num}_Trust',
            f'Opt{target_option_num}_Modernity',
            f'Opt{target_option_num}_Premium_Feel',
        ]

    categorical_cols = [c for c in ['Cat_Breed_Cat', 'Brand_Std'] if c in df_in.columns]
    numeric_cols = [c for c in numeric_cols + option_feature_cols if c in df_in.columns]

    X_num = df_in[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    if categorical_cols:
        X_cat = pd.get_dummies(df_in[categorical_cols].fillna('Unknown'), prefix=categorical_cols)
        X = pd.concat([X_num, X_cat], axis=1)
        feature_cols = numeric_cols + list(X_cat.columns)
    else:
        X = X_num
        feature_cols = numeric_cols
    return X, feature_cols


# ============================================================
# PHASE 1: DATA PROFILING (PARALLEL)
# ============================================================
def run_data_profiling(df_raw: pd.DataFrame) -> dict:
    """Run statistical profiling in parallel using ThreadPoolExecutor"""
    logging.info("=" * 60)
    logging.info("PHASE 1: DATA PROFILING (Parallel)")
    logging.info("=" * 60)

    def get_summary():
        return df_raw.describe(include='all')

    def get_missing():
        return df_raw.isnull().sum()

    def get_dtypes():
        return df_raw.dtypes

    def get_numeric_corr():
        numeric_df = df_raw.select_dtypes(include=[np.number])
        return numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

    def get_value_counts(col):
        return df_raw[col].value_counts()

    t0 = time.time()

    # Run independent profilers in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_summary = executor.submit(get_summary)
        f_missing = executor.submit(get_missing)
        f_dtypes = executor.submit(get_dtypes)
        f_corr = executor.submit(get_numeric_corr)

        results = {
            'summary': f_summary.result(),
            'missing': f_missing.result(),
            'dtypes': f_dtypes.result(),
            'correlation': f_corr.result()
        }

    # Run value counts for key columns in parallel
    key_cols = ['Experience', 'Cat_Breed', 'Current_Brand', 'Age', 'Gender']
    key_cols = [c for c in key_cols if c in df_raw.columns]
    if key_cols:
        with ThreadPoolExecutor(max_workers=len(key_cols)) as executor:
            value_count_futures = {col: executor.submit(get_value_counts, col) for col in key_cols}
            results['value_counts'] = {col: f.result() for col, f in value_count_futures.items()}
    else:
        results['value_counts'] = {}

    logging.info(f"[Profiling] Completed in {time.time() - t0:.2f}s")
    logging.info(f"[Profiling] Raw shape: {df_raw.shape}")
    logging.info(f"[Profiling] Missing values per column:\n{results['missing'].to_string()}")

    return results


# ============================================================
# LOAD CONFIG
# ============================================================
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info("Configuration loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load config.json: {e}")
    sys.exit(1)

# ============================================================
# LOAD DATA
# ============================================================
logging.info("=" * 60)
logging.info("LOADING DATA...")
logging.info("=" * 60)

CSV_FILE = 'BU Data from Survey Cases_final(5).csv'
try:
    df_raw = pd.read_csv(CSV_FILE, encoding='utf-8-sig', header=1)
    logging.info(f"Raw shape: {df_raw.shape}")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    sys.exit(1)

# ============================================================
# PHASE 1: RUN DATA PROFILING
# ============================================================
profile_results = run_data_profiling(df_raw)

# ============================================================
# RENAME COLUMNS (ชื่อคอลัมน์ภาษาไทยถูก corrupt เป็น ?)
# ============================================================
col_names = {int(k): v for k, v in config['col_names'].items()}

# Option 1-10, each has 5 sub-attributes
option_attrs = ['Attractiveness', 'Trust', 'Modernity', 'Premium_Feel', 'Purchase_Intent']
for opt in range(1, 11):
    for j, attr in enumerate(option_attrs):
        col_idx = 22 + (opt - 1) * 5 + j
        col_names[col_idx] = f'Opt{opt}_{attr}'

df = df_raw.copy()
df.columns = [col_names.get(i, f'col_{i}') for i in range(len(df.columns))]
logging.info(f"Columns renamed: {len(df.columns)} columns")

# ============================================================
# STAGE 1: TARGET VARIABLE IDENTIFICATION
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 1: TARGET VARIABLE IDENTIFICATION")
logging.info("=" * 60)
t_stage1 = time.time()

df['Target_Option'] = df['Top3_Choices'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) else np.nan
)
logging.info(f"Target_Option created. Distribution:\n{df['Target_Option'].value_counts().to_string()}")
logging.info(f"[Diagnostic] Stage 1 completed in {time.time() - t_stage1:.2f}s | Rows: {len(df)}")

# ============================================================
# STAGE 2: DATA CLEANING
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 2: DATA CLEANING")
logging.info("=" * 60)
t_stage2 = time.time()

# 2a. Drop rows that are completely NaN (except Timestamp/Experience)
important_cols = ['Target_Option', 'Age', 'Gender']
df_clean = df.dropna(subset=important_cols)
logging.info(f"After dropping NaN in key columns: {len(df_clean)} rows (from {len(df)})")

# 2b. Handle Experience filter (Col 1)
df_clean = df_clean[df_clean['Experience'] == 'เคย']
df_clean['Experience'] = 'Yes'
if len(df_clean) != 148:
    logging.warning(f"Analytic sample size is {len(df_clean)} rows, expected 148 based on current survey branch.")

# 2c. Standardize Cat Breed (Col 3)
purebreds = config['purebred_keywords']
def categorize_breed(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    for breed in purebreds:
        if breed in text:
            return 'Purebred'
    return 'Mixed/Stray'

df_clean['Cat_Breed_Cat'] = df_clean['Cat_Breed'].apply(categorize_breed)
logging.info(f"Cat Breed categories: {df_clean['Cat_Breed_Cat'].value_counts().to_dict()}")

# 2d. Standardize Current Brand (Col 4)
brand_map = config['brand_map']
def standardize_brand(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    for keyword, std_name in brand_map.items():
        if keyword in text:
            return std_name
    return 'Other'

df_clean['Brand_Std'] = df_clean['Current_Brand'].apply(standardize_brand)
logging.info(f"Brand categories: {df_clean['Brand_Std'].value_counts().head(10).to_dict()}")

logging.info(f"[Diagnostic] Stage 2 completed in {time.time() - t_stage2:.2f}s | Cleaned rows: {len(df_clean)}")

# ============================================================
# STAGE 3: FEATURE SELECTION & ENCODING
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 3: FEATURE SELECTION & ENCODING")
logging.info("=" * 60)
t_stage3 = time.time()

# 3a. Likert Scale -> Numeric
def map_likert_5(val):
    if pd.isna(val):
        return np.nan
    mapping = {'มากที่สุด': 5, 'มาก': 4, 'ปานกลาง': 3, 'น้อย': 2, 'น้อยที่สุด': 1}
    return mapping.get(str(val).strip(), np.nan)

likert5_cols = [
    'Factor_Taste', 'Factor_Brand_Rep', 'Factor_Price',
    'Factor_Ingredients', 'Factor_Packaging',
    'PkgFactor_Taste', 'PkgFactor_Freshness', 'PkgFactor_Nutrition',
    'PkgFactor_Safety', 'PkgFactor_Convenience', 'PkgFactor_Promotion',
    'PkgFactor_Quality', 'PkgFactor_Price'
]
for col in likert5_cols:
    df_clean[col] = df_clean[col].apply(map_likert_5)
logging.info(f"Likert-5 encoded: {len(likert5_cols)} columns")

# 3b. Option Ratings -> Numeric
def map_likert_4(val):
    if pd.isna(val):
        return np.nan
    mapping = {'เห็นด้วยที่สุด': 4, 'เห็นด้วยอย่างยิ่ง': 4, 'เห็นด้วย': 3, 'เฉยๆ': 2, 'ไม่เห็นด้วย': 1, 'ไม่เห็นด้วยเลย': 1}
    return mapping.get(str(val).strip(), np.nan)

option_cols = [c for c in df_clean.columns if c.startswith('Opt')]
for col in option_cols:
    df_clean[col] = df_clean[col].apply(map_likert_4)
logging.info(f"Option ratings encoded: {len(option_cols)} columns")

# 3c. Ordinal Encoding for Age
age_map = {}
for val in df_clean['Age'].dropna().unique():
    s = str(val)
    if '20-29' in s:
        age_map[val] = 1
    elif '30-39' in s:
        age_map[val] = 2
    elif '40-49' in s:
        age_map[val] = 3
    else:
        age_map[val] = 4
df_clean['Age_Ordinal'] = df_clean['Age'].map(age_map)
logging.info(f"Age ordinal encoded")

# 3d. One-Hot Encoding for Gender
gender_map = {'ชาย': 'Male', 'หญิง': 'Female', 'อื่นๆ': 'Other'}
df_clean['Gender_Label'] = df_clean['Gender'].map(gender_map).fillna('Other')
gender_dummies = pd.get_dummies(df_clean['Gender_Label'], prefix='Gender')
df_clean = pd.concat([df_clean, gender_dummies], axis=1)
gender_feature_cols = gender_dummies.columns.tolist()

# 3e. One-Hot Encoding for Marital Status
marital_map = {
    'โสด ไม่มีแฟน': 'Single',
    'มีแฟนแต่ยังไม่แต่งงาน': 'In_Relationship',
    'แต่งงานแล้ว': 'Married',
    'หย่าร้าง/เป็นม่าย': 'Divorced'
}
df_clean['Marital_Label'] = df_clean['Marital_Status'].map(marital_map).fillna('Other')
marital_dummies = pd.get_dummies(df_clean['Marital_Label'], prefix='Marital')
df_clean = pd.concat([df_clean, marital_dummies], axis=1)
marital_feature_cols = marital_dummies.columns.tolist()
logging.info(f"One-Hot encoded Gender & Marital Status")

logging.info(f"[Diagnostic] Encoding completed in {time.time() - t_stage3:.2f}s")

# ============================================================
# STAGE 3B: PARALLEL FEATURE SELECTION (ANOVA + RF)
# ============================================================
logging.info("\n--- Feature Selection: Running ANOVA and RandomForest in parallel ---")
t_fselect = time.time()

# Prepare feature columns
feature_cols = likert5_cols + option_cols + ['Age_Ordinal']
feature_cols += gender_feature_cols + marital_feature_cols
valid_mask = df_clean['Target_Option'].notna()

def run_anova_feature_selection():
    """ANOVA F-test feature selection"""
    scores = {}
    for col in feature_cols:
        if col not in df_clean.columns:
            continue
        col_valid = df_clean.loc[valid_mask, col].notna()
        if col_valid.sum() < 10:
            continue
        groups = []
        for opt in df_clean.loc[valid_mask, 'Target_Option'].unique():
            group_data = df_clean.loc[valid_mask & (df_clean['Target_Option'] == opt), col].dropna()
            if len(group_data) >= 2:
                groups.append(group_data.values)
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                scores[col] = {'F_stat': f_stat, 'p_value': p_val}
            except:
                pass
    scores_df = pd.DataFrame(scores).T.sort_values('p_value')
    return scores_df.head(10)

def run_random_forest_feature_selection():
    """Random Forest feature importance selection"""
    rf_data = df_clean.dropna(subset=['Target_Option']).copy()
    exclude_cols = ['Gender_Label', 'Marital_Label', 'Marital_Status']
    rf_features = likert5_cols + option_cols + ['Age_Ordinal'] + gender_feature_cols + marital_feature_cols
    rf_features = [c for c in rf_features if c in rf_data.columns]

    X = rf_data[rf_features].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(rf_data['Target_Option'])

    k_best = min(20, len(rf_features))
    selector = SelectKBest(score_func=chi2, k=k_best)
    X_kbest = selector.fit_transform(X, y)

    selected_mask = selector.get_support()
    selected_features = [rf_features[i] for i in range(len(rf_features)) if selected_mask[i]]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_kbest, y)

    rf_scores = pd.DataFrame({'Feature': selected_features, 'Importance': rf.feature_importances_})
    return rf_scores.sort_values('Importance', ascending=False)

# Run ANOVA and RandomForest in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    f_anova = executor.submit(run_anova_feature_selection)
    f_rf = executor.submit(run_random_forest_feature_selection)

    anova_result = f_anova.result()
    rf_result = f_rf.result()

top10_features = anova_result.index.tolist()
logging.info(f"\n[ANOVA] Top 10 Features:\n{anova_result.to_string()}")
logging.info(f"\n[RandomForest] Top 10 Features:\n{rf_result.head(10).to_string(index=False)}")
logging.info(f"[Diagnostic] Parallel feature selection completed in {time.time() - t_fselect:.2f}s")

# ============================================================
# STAGE 3C: TEXT MINING & INSIGHT EXTRACTION
# ============================================================
logging.info("\n--- Insight Extraction (Text Mining) ---")
insight_cols = []
insights_map = config['insights_map']

df_clean['Combined_Text'] = df_clean['Packaging_Suggestion'].fillna('') + ' ' + df_clean['Current_Brand_Detail'].fillna('')

for col_name, keywords in insights_map.items():
    pattern = '|'.join(keywords)
    df_clean[col_name] = df_clean['Combined_Text'].str.contains(pattern, case=False, na=False).astype(int)
    insight_cols.append(col_name)

logging.info(f"Extracted {len(insight_cols)} insight features from open-ended questions.")

# ============================================================
# STAGE 3D: PURCHASE INTENT TARGET PREPARATION & SPARSITY REDUCTION
# ============================================================
logging.info("\n--- Purchase Intent Target Preparation ---")

purchase_cfg = config.get('purchase_intent', {})
core_options = purchase_cfg.get('core_options', [3, 6])
intent_threshold = purchase_cfg.get('threshold', 3)
min_class_size = purchase_cfg.get('min_class_size_for_grouping', 5)

target_counts = df_clean['Target_Option'].value_counts(dropna=True)
rare_classes = target_counts[target_counts < min_class_size].index.tolist()
df_clean['Target_Option_Grouped'] = df_clean['Target_Option'].apply(
    lambda x: 'Other' if x in rare_classes else x
)
logging.info(f"Target_Option_Grouped distribution:\n{df_clean['Target_Option_Grouped'].value_counts().to_string()}")

for opt_num in core_options:
    pi_col = f'Opt{opt_num}_Purchase_Intent'
    target_col = f'Opt{opt_num}_High_Intent'
    if pi_col in df_clean.columns:
        df_clean[target_col] = (df_clean[pi_col] >= intent_threshold).astype(int)
        logging.info(f"Created {target_col} using threshold >= {intent_threshold}")

_zero_series = pd.Series(0, index=df_clean.index)
df_clean['Design_Special'] = (
    (df_clean.get('Design_Matte', _zero_series) == 1) | (df_clean.get('Design_Cartoon', _zero_series) == 1)
).astype(int)
df_clean['Need_Functional_Pkg'] = (
    (df_clean.get('Need_Small_Packs', _zero_series) == 1) |
    (df_clean.get('Need_Pour_Lid', _zero_series) == 1) |
    (df_clean.get('Need_Clear_Window', _zero_series) == 1)
).astype(int)

final_insight_cols = [
    c for c in ['Need_Big_Text', 'Need_Interactive', 'Need_Sodium_Info', 'Design_Special', 'Need_Functional_Pkg']
    if c in df_clean.columns
]

if 'Need_Topping' in df_clean.columns:
    df_clean.drop(columns=['Need_Topping'], inplace=True)
    insight_cols = [c for c in insight_cols if c != 'Need_Topping']
    logging.info("Dropped sparse feature Need_Topping")

logging.info(f"Final grouped insight columns: {final_insight_cols}")

# ============================================================
# STAGE 6: PURCHASE INTENT CLASSIFICATION
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 6: PURCHASE INTENT CLASSIFICATION")
logging.info("=" * 60)
t_model = time.time()

model_results_records = []
imbalance_records = []
binary_model_artifacts = {}
preference_model_artifact = {}

# --- Model 1: Preference (Multiclass) ---
pref_df = df_clean.dropna(subset=['Target_Option_Grouped']).copy()
X_pref, pref_feature_cols = _build_feature_frame(pref_df, target_option_num=None)
y_pref = pref_df['Target_Option_Grouped'].astype(str).reset_index(drop=True)

pref_class_counts = y_pref.value_counts()
pref_min_class = int(pref_class_counts.min()) if not pref_class_counts.empty else 0
pref_splits = max(2, min(int(purchase_cfg.get('cv_splits', 5)), pref_min_class if pref_min_class else 2))
pref_repeats = int(purchase_cfg.get('cv_repeats', 3))
pref_cv = RepeatedStratifiedKFold(n_splits=pref_splits, n_repeats=pref_repeats, random_state=42)

rf_pref = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
)

pref_eval = _build_oof_predictions(rf_pref, X_pref, y_pref, pref_cv, threshold=0.5, oversample=False)
pref_report = _safe_classification_report(pref_eval['y_true'], pref_eval['y_pred'])
pref_macro_f1 = f1_score(pref_eval['y_true'], pref_eval['y_pred'], average='macro', zero_division=0)
pref_bal_acc = balanced_accuracy_score(pref_eval['y_true'], pref_eval['y_pred'])
pref_acc = accuracy_score(pref_eval['y_true'], pref_eval['y_pred'])

pref_final_model = clone(rf_pref)
pref_final_model.fit(X_pref, y_pref)

preference_model_artifact = {
    'features': pref_feature_cols,
    'model': pref_final_model,
    'cv': pref_cv,
    'eval': pref_eval,
    'report': pref_report,
    'metrics': {
        'macro_f1': pref_macro_f1,
        'balanced_accuracy': pref_bal_acc,
        'accuracy': pref_acc,
    },
}

model_results_records.extend(
    _flatten_classification_report(pref_report, 'Preference', 'RandomForest', 'RepeatedCV', threshold=None)
)
model_results_records.append({
    'Target': 'Preference',
    'Model': 'RandomForest',
    'Step': 'RepeatedCV_Summary',
    'Label': '__summary__',
    'Threshold': None,
    'Precision': None,
    'Recall': pref_bal_acc,
    'F1': pref_macro_f1,
    'Support': int(len(y_pref)),
})

# --- Model 2: Purchase Intent (Binary) ---
for opt_num in core_options:
    target_col = f'Opt{opt_num}_High_Intent'
    if target_col not in df_clean.columns:
        logging.warning(f"Skipping {target_col}: column not found")
        continue

    intent_df = df_clean.dropna(subset=[target_col]).copy()
    X_intent, intent_feature_cols = _build_feature_frame(intent_df, target_option_num=opt_num)
    y_intent = intent_df[target_col].astype(int).reset_index(drop=True)

    intent_counts = y_intent.value_counts()
    intent_min_class = int(intent_counts.min()) if not intent_counts.empty else 0
    intent_splits = max(2, min(int(purchase_cfg.get('cv_splits', 5)), intent_min_class if intent_min_class else 2))
    intent_repeats = int(purchase_cfg.get('cv_repeats', 3))
    intent_cv = RepeatedStratifiedKFold(n_splits=intent_splits, n_repeats=intent_repeats, random_state=42)

    baseline_estimator = RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    balanced_estimator = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )

    baseline_eval = _build_oof_predictions(baseline_estimator, X_intent, y_intent, intent_cv, threshold=0.5, positive_label=1, oversample=False)
    balanced_eval = _build_oof_predictions(balanced_estimator, X_intent, y_intent, intent_cv, threshold=0.5, positive_label=1, oversample=False)
    tuned_threshold, pr_data = _best_threshold_from_pr(balanced_eval['y_true'], balanced_eval['y_score'])
    tuned_pred = (balanced_eval['y_score'] >= tuned_threshold).astype(int)
    tuned_report = _safe_classification_report(balanced_eval['y_true'], tuned_pred, labels=[0, 1], target_names=['Low', 'High'])

    oversample_eval = _build_oof_predictions(baseline_estimator, X_intent, y_intent, intent_cv, threshold=0.5, positive_label=1, oversample=True)
    oversample_report = _safe_classification_report(oversample_eval['y_true'], oversample_eval['y_pred'], labels=[0, 1], target_names=['Low', 'High'])

    final_model = clone(balanced_estimator)
    final_model.fit(X_intent, y_intent)

    average_precision = average_precision_score(balanced_eval['y_true'], balanced_eval['y_score'])
    best_f1 = f1_score(balanced_eval['y_true'], tuned_pred, zero_division=0)
    best_bal_acc = balanced_accuracy_score(balanced_eval['y_true'], tuned_pred)
    best_acc = accuracy_score(balanced_eval['y_true'], tuned_pred)

    binary_model_artifacts[opt_num] = {
        'target_col': target_col,
        'feature_cols': intent_feature_cols,
        'model': final_model,
        'cv': intent_cv,
        'baseline_eval': baseline_eval,
        'balanced_eval': balanced_eval,
        'oversample_eval': oversample_eval,
        'threshold': tuned_threshold,
        'pr_data': pr_data,
        'average_precision': average_precision,
        'feature_importances': pd.Series(final_model.feature_importances_, index=intent_feature_cols).sort_values(ascending=False),
        'reports': {
            'baseline': _safe_classification_report(baseline_eval['y_true'], baseline_eval['y_pred'], labels=[0, 1], target_names=['Low', 'High']),
            'balanced': _safe_classification_report(balanced_eval['y_true'], balanced_eval['y_pred'], labels=[0, 1], target_names=['Low', 'High']),
            'tuned': tuned_report,
            'oversample': oversample_report,
        },
        'metrics': {
            'baseline': {
                'f1': f1_score(baseline_eval['y_true'], baseline_eval['y_pred'], zero_division=0),
                'balanced_acc': balanced_accuracy_score(baseline_eval['y_true'], baseline_eval['y_pred']),
                'pr_auc': average_precision_score(baseline_eval['y_true'], baseline_eval['y_score']),
                'accuracy': accuracy_score(baseline_eval['y_true'], baseline_eval['y_pred']),
            },
            'balanced': {
                'f1': f1_score(balanced_eval['y_true'], balanced_eval['y_pred'], zero_division=0),
                'balanced_acc': balanced_accuracy_score(balanced_eval['y_true'], balanced_eval['y_pred']),
                'pr_auc': average_precision_score(balanced_eval['y_true'], balanced_eval['y_score']),
                'accuracy': accuracy_score(balanced_eval['y_true'], balanced_eval['y_pred']),
            },
            'tuned': {
                'f1': best_f1,
                'balanced_acc': best_bal_acc,
                'pr_auc': average_precision,
                'accuracy': best_acc,
            },
            'oversample': {
                'f1': f1_score(oversample_eval['y_true'], oversample_eval['y_pred'], zero_division=0),
                'balanced_acc': balanced_accuracy_score(oversample_eval['y_true'], oversample_eval['y_pred']),
                'pr_auc': average_precision_score(oversample_eval['y_true'], oversample_eval['y_score']),
                'accuracy': accuracy_score(oversample_eval['y_true'], oversample_eval['y_pred']),
            },
        }
    }

    # Model results export rows
    for step_name, report in binary_model_artifacts[opt_num]['reports'].items():
        model_results_records.extend(
            _flatten_classification_report(report, f'Opt{opt_num}_High_Intent', 'RandomForest', step_name, threshold=tuned_threshold if step_name == 'tuned' else (0.5 if step_name != 'oversample' else 0.5))
        )

    model_results_records.append({
        'Target': f'Opt{opt_num}_High_Intent',
        'Model': 'RandomForest',
        'Step': 'Summary',
        'Label': '__summary__',
        'Threshold': tuned_threshold,
        'Precision': None,
        'Recall': best_bal_acc,
        'F1': best_f1,
        'Support': int(len(y_intent)),
    })

    imbalance_records.extend([
        {
            'Target': f'Opt{opt_num}_High_Intent',
            'Step': '1_Baseline',
            'F1': binary_model_artifacts[opt_num]['metrics']['baseline']['f1'],
            'Balanced_Accuracy': binary_model_artifacts[opt_num]['metrics']['baseline']['balanced_acc'],
            'PR_AUC': binary_model_artifacts[opt_num]['metrics']['baseline']['pr_auc'],
            'Threshold': 0.5,
        },
        {
            'Target': f'Opt{opt_num}_High_Intent',
            'Step': '2_Class_Weight',
            'F1': binary_model_artifacts[opt_num]['metrics']['balanced']['f1'],
            'Balanced_Accuracy': binary_model_artifacts[opt_num]['metrics']['balanced']['balanced_acc'],
            'PR_AUC': binary_model_artifacts[opt_num]['metrics']['balanced']['pr_auc'],
            'Threshold': 0.5,
        },
        {
            'Target': f'Opt{opt_num}_High_Intent',
            'Step': '3_Threshold_Tuned',
            'F1': binary_model_artifacts[opt_num]['metrics']['tuned']['f1'],
            'Balanced_Accuracy': binary_model_artifacts[opt_num]['metrics']['tuned']['balanced_acc'],
            'PR_AUC': binary_model_artifacts[opt_num]['metrics']['tuned']['pr_auc'],
            'Threshold': tuned_threshold,
        },
        {
            'Target': f'Opt{opt_num}_High_Intent',
            'Step': '4_Oversample',
            'F1': binary_model_artifacts[opt_num]['metrics']['oversample']['f1'],
            'Balanced_Accuracy': binary_model_artifacts[opt_num]['metrics']['oversample']['balanced_acc'],
            'PR_AUC': binary_model_artifacts[opt_num]['metrics']['oversample']['pr_auc'],
            'Threshold': 0.5,
        },
    ])

    logging.info(
        f"[Opt{opt_num}] baseline F1={binary_model_artifacts[opt_num]['metrics']['baseline']['f1']:.3f}, "
        f"balanced F1={binary_model_artifacts[opt_num]['metrics']['balanced']['f1']:.3f}, "
        f"tuned F1={best_f1:.3f} @ threshold={tuned_threshold:.3f}, "
        f"oversample F1={binary_model_artifacts[opt_num]['metrics']['oversample']['f1']:.3f}"
    )

model_results_df = pd.DataFrame(model_results_records)
imbalance_comparison_df = pd.DataFrame(imbalance_records)

logging.info(f"[Diagnostic] Purchase intent models completed in {time.time() - t_model:.2f}s")

# ============================================================
# STAGE 4: DATA VISUALIZATION (PARALLEL)
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 4: DATA VISUALIZATION (Parallel Chart Generation)")
logging.info("=" * 60)
t_viz = time.time()

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

# Thread-safe chart generation using Figure directly (not global plt state)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def generate_chart_demographics():
    """Chart 1: Demographic Profile"""
    fig = Figure(figsize=(18, 5))
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.suptitle('Demographic Profile of Respondents', fontsize=16, fontweight='bold')

    age_counts = df_clean['Age_Ordinal'].value_counts().sort_index()
    age_labels = ['20-29', '30-39', '40-49', '50+']
    axes[0].bar(age_labels[:len(age_counts)], age_counts.values, color=COLORS[:len(age_counts)], edgecolor='white')
    axes[0].set_title('Age Distribution', fontsize=13)

    gender_counts = df_clean['Gender_Label'].value_counts()
    axes[1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=COLORS[:len(gender_counts)], startangle=90)
    axes[1].set_title('Gender Distribution', fontsize=13)

    marital_counts = df_clean['Marital_Label'].value_counts()
    axes[2].barh(marital_counts.index, marital_counts.values, color=COLORS[:len(marital_counts)], edgecolor='white')
    axes[2].set_title('Marital Status', fontsize=13)

    fig.tight_layout()
    canvas.print_figure('chart1_demographics.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 1 saved: chart1_demographics.png")

def generate_chart_target():
    """Chart 2: Target Variable Distribution"""
    fig = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    target_counts = df_clean['Target_Option'].value_counts().sort_index()
    ax.bar(target_counts.index, target_counts.values, color=COLORS[:len(target_counts)], edgecolor='white')
    ax.set_title('Distribution of Top 1 Packaging Choice (Target Variable)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    canvas.print_figure('chart2_target_distribution.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 2 saved: chart2_target_distribution.png")

def generate_chart_correlation():
    """Chart 3: Correlation Heatmap"""
    heatmap_cols = [c for c in top10_features if c in df_clean.columns][:10]
    if len(heatmap_cols) >= 2:
        corr_data = df_clean[heatmap_cols].dropna()
        if len(corr_data) > 5:
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvasAgg(fig)
            ax = fig.subplots()
            corr_matrix = corr_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax, linewidths=0.5, square=True)
            ax.set_title('Correlation Heatmap: Top 10 Features', fontsize=14, fontweight='bold')
            fig.tight_layout()
            canvas.print_figure('chart3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
            fig.clf()
            logging.info("Chart 3 saved: chart3_correlation_heatmap.png")

def generate_chart_mean_scores():
    """Chart 4: Mean Option Scores"""
    opt_means = {}
    for opt in range(1, 11):
        opt_cols_i = [c for c in df_clean.columns if c.startswith(f'Opt{opt}_')]
        if opt_cols_i:
            opt_means[f'Option {opt}'] = df_clean[opt_cols_i].mean().mean()

    if opt_means:
        fig = Figure(figsize=(12, 6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        opts = list(opt_means.keys())
        vals = list(opt_means.values())
        ax.bar(opts, vals, color=COLORS[:len(opts)], edgecolor='white')
        ax.set_title('Mean Rating Score by Packaging Option', fontsize=14, fontweight='bold')
        fig.tight_layout()
        canvas.print_figure('chart4_mean_option_scores.png', dpi=150, bbox_inches='tight')
        fig.clf()
        logging.info("Chart 4 saved: chart4_mean_option_scores.png")

def generate_chart_insights():
    """Chart 5: Qualitative Insights Distribution"""
    if insight_cols:
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        insight_counts = df_clean[insight_cols].sum().sort_values(ascending=True)
        ax.barh(insight_counts.index, insight_counts.values, color=COLORS[0], edgecolor='white')
        ax.set_title('Top Requested Features & Pain Points (from Text)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        canvas.print_figure('chart5_insights_distribution.png', dpi=150, bbox_inches='tight')
        fig.clf()
        logging.info("Chart 5 saved: chart5_insights_distribution.png")

def generate_chart_insights_vs_option():
    """Chart 6: Top Insights vs Target Option"""
    if insight_cols:
        top_insights = df_clean[insight_cols].sum().sort_values(ascending=True).tail(5).index.tolist()
        plot_data = pd.DataFrame()
        for col in top_insights:
            counts = df_clean[df_clean[col] == 1].groupby('Target_Option').size()
            plot_data[col] = counts
        plot_data = plot_data.fillna(0)

        if not plot_data.empty:
            fig = Figure(figsize=(12, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.subplots()
            plot_data.T.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
            ax.set_title('Top 5 Requested Features by Target Packaging Option', fontsize=14, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()
            canvas.print_figure('chart6_insights_vs_option.png', dpi=150, bbox_inches='tight')
            fig.clf()
            logging.info("Chart 6 saved: chart6_insights_vs_option.png")

def generate_chart_rf_importance():
    """Chart 7: Random Forest Feature Importance"""
    if 'rf_result' in dir() and not rf_result.empty:
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        sns.barplot(x='Importance', y='Feature', data=rf_result.head(10), palette='viridis', ax=ax)
        ax.set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('Features')
        fig.tight_layout()
        canvas.print_figure('chart7_rf_feature_importance.png', dpi=150, bbox_inches='tight')
        fig.clf()
        logging.info("Chart 7 saved: chart7_rf_feature_importance.png")

def generate_chart_purchase_intent_distribution():
    """Chart 8: Purchase intent score distribution for Opt3 vs Opt6"""
    available_opts = [opt for opt in core_options if f'Opt{opt}_Purchase_Intent' in df_clean.columns]
    if len(available_opts) < 2:
        return

    fig = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()

    score_levels = [1, 2, 3, 4]
    plot_df = pd.DataFrame(index=score_levels)
    for opt in available_opts:
        col = f'Opt{opt}_Purchase_Intent'
        counts = df_clean[col].value_counts().reindex(score_levels, fill_value=0)
        plot_df[f'Opt{opt}'] = counts

    plot_df.plot(kind='bar', ax=ax, color=[COLORS[0], COLORS[2]])
    ax.set_title('Purchase Intent Distribution: Opt3 vs Opt6', fontsize=14, fontweight='bold')
    ax.set_xlabel('Purchase Intent Score')
    ax.set_ylabel('Count')
    ax.set_xticklabels([str(i) for i in score_levels], rotation=0)
    ax.legend(title='Option')
    fig.tight_layout()
    canvas.print_figure('chart8_purchase_intent_dist.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 8 saved: chart8_purchase_intent_dist.png")

def generate_chart_pr_curve():
    """Chart 9: Precision-Recall curve for the core intent model"""
    core_target = core_options[0] if core_options else None
    if core_target not in binary_model_artifacts:
        return

    artifact = binary_model_artifacts[core_target]
    y_true = artifact['balanced_eval']['y_true']
    y_score = artifact['balanced_eval']['y_score']
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = artifact['average_precision']

    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(recall, precision, color=COLORS[0], linewidth=2)
    ax.set_title(f'Precision-Recall Curve for Opt{core_target}_High_Intent (AP={ap:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.axhline(y=y_true.mean(), color='gray', linestyle='--', linewidth=1, label='Baseline')
    ax.legend()
    fig.tight_layout()
    canvas.print_figure('chart9_pr_auc_curve.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 9 saved: chart9_pr_auc_curve.png")

def generate_chart_intent_feature_importance():
    """Chart 10: Feature importance for the core intent model"""
    core_target = core_options[0] if core_options else None
    if core_target not in binary_model_artifacts:
        return

    importance_series = binary_model_artifacts[core_target]['feature_importances'].head(10)
    if importance_series.empty:
        return

    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    plot_df = importance_series.reset_index()
    plot_df.columns = ['Feature', 'Importance']
    sns.barplot(data=plot_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title(f'Top Feature Importance for Opt{core_target}_High_Intent', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    canvas.print_figure('chart10_intent_feature_importance.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 10 saved: chart10_intent_feature_importance.png")

def generate_chart_imbalance_comparison():
    """Chart 11: Imbalance ladder comparison for the core intent model"""
    core_target = core_options[0] if core_options else None
    if core_target is None:
        return
    target_name = f'Opt{core_target}_High_Intent'
    plot_df = imbalance_comparison_df[imbalance_comparison_df['Target'] == target_name].copy()
    if plot_df.empty:
        return

    fig = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    plot_long = plot_df.melt(id_vars=['Step'], value_vars=['F1', 'Balanced_Accuracy', 'PR_AUC'], var_name='Metric', value_name='Score')
    step_order = ['1_Baseline', '2_Class_Weight', '3_Threshold_Tuned', '4_Oversample']
    sns.barplot(data=plot_long, x='Step', y='Score', hue='Metric', order=step_order, ax=ax)
    ax.set_title(f'Imbalance Ladder Comparison for Opt{core_target}_High_Intent', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    canvas.print_figure('chart11_imbalance_comparison.png', dpi=150, bbox_inches='tight')
    fig.clf()
    logging.info("Chart 11 saved: chart11_imbalance_comparison.png")

# Define chart generation functions
chart_generators = [
    generate_chart_demographics,
    generate_chart_target,
    generate_chart_correlation,
    generate_chart_mean_scores,
    generate_chart_insights,
    generate_chart_insights_vs_option,
    generate_chart_rf_importance,
    generate_chart_purchase_intent_distribution,
    generate_chart_pr_curve,
    generate_chart_intent_feature_importance,
    generate_chart_imbalance_comparison,
]

# Run all charts in parallel
with ThreadPoolExecutor(max_workers=len(chart_generators)) as executor:
    futures = [executor.submit(gen) for gen in chart_generators]
    for f in as_completed(futures):
        try:
            f.result()
        except Exception as e:
            logging.warning(f"Chart generation failed: {e}")

logging.info(f"[Diagnostic] All charts generated in parallel in {time.time() - t_viz:.2f}s")

# ============================================================
# STAGE 5: EXPORT
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 5: EXPORT")
logging.info("=" * 60)
t_export = time.time()

# 5a. Export CSV
try:
    export_cols = (
        ['Target_Option', 'Target_Option_Grouped', 'Experience', 'Cat_Breed_Cat', 'Brand_Std']
        + likert5_cols + option_cols
        + [f'Opt{opt}_High_Intent' for opt in core_options if f'Opt{opt}_High_Intent' in df_clean.columns]
        + ['Age_Ordinal', 'Gender_Label', 'Marital_Label']
        + gender_feature_cols + marital_feature_cols
        + insight_cols
        + final_insight_cols
    )
    export_cols = [c for c in export_cols if c in df_clean.columns]
    df_export = df_clean[export_cols].copy()
    df_export.to_csv('cleaned_survey_data.csv', index=False, encoding='utf-8-sig')
    logging.info(f"Exported: cleaned_survey_data.csv ({df_export.shape})")
except Exception as e:
    logging.error(f"Failed to export CSV: {e}")

try:
    if 'model_results_df' in globals() and not model_results_df.empty:
        model_results_df.to_csv('model_results.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Exported: model_results.csv ({model_results_df.shape})")
except Exception as e:
    logging.error(f"Failed to export model_results.csv: {e}")

try:
    if 'imbalance_comparison_df' in globals() and not imbalance_comparison_df.empty:
        imbalance_comparison_df.to_csv('imbalance_comparison.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Exported: imbalance_comparison.csv ({imbalance_comparison_df.shape})")
except Exception as e:
    logging.error(f"Failed to export imbalance_comparison.csv: {e}")
