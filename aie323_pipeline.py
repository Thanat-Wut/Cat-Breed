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
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import logging
import sys
import time

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
logging.info(f"One-Hot encoded Gender & Marital Status")

logging.info(f"[Diagnostic] Encoding completed in {time.time() - t_stage3:.2f}s")

# ============================================================
# STAGE 3B: PARALLEL FEATURE SELECTION (ANOVA + RF)
# ============================================================
logging.info("\n--- Feature Selection: Running ANOVA and RandomForest in parallel ---")
t_fselect = time.time()

# Prepare feature columns
feature_cols = likert5_cols + option_cols + ['Age_Ordinal']
feature_cols += [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]
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
    rf_features = likert5_cols + option_cols + ['Age_Ordinal'] + [c for c in df_clean.columns if (c.startswith('Gender_') or c.startswith('Marital_')) and c not in exclude_cols]
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

# Define chart generation functions
chart_generators = [
    generate_chart_demographics,
    generate_chart_target,
    generate_chart_correlation,
    generate_chart_mean_scores,
    generate_chart_insights,
    generate_chart_insights_vs_option,
    generate_chart_rf_importance,
]

# Run all charts in parallel
with ThreadPoolExecutor(max_workers=7) as executor:
    futures = [executor.submit(gen) for gen in chart_generators]
    for f in as_completed(futures):
        try:
            f.result()
        except Exception as e:
            logging.warning(f"Chart generation failed: {e}")

logging.info(f"[Diagnostic] All charts generated in parallel in {time.time() - t_viz:.2f}s")

# ============================================================
# STAGE 5: EXPORT & SUMMARY
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 5: EXPORT & SUMMARY")
logging.info("=" * 60)
t_export = time.time()

# 5a. Export CSV
try:
    export_cols = (
        ['Target_Option', 'Experience', 'Cat_Breed_Cat', 'Brand_Std']
        + likert5_cols + option_cols
        + ['Age_Ordinal', 'Gender_Label', 'Marital_Label']
        + [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]
        + insight_cols
    )
    export_cols = [c for c in export_cols if c in df_clean.columns]
    df_export = df_clean[export_cols].copy()
    df_export.to_csv('cleaned_survey_data.csv', index=False, encoding='utf-8-sig')
    logging.info(f"Exported: cleaned_survey_data.csv ({df_export.shape})")
except Exception as e:
    logging.error(f"Failed to export CSV: {e}")

# 5b. Summary Text
try:
    summary_text = f"""=== Logic for Target & Feature Selection ===
1. Target Variable: Extract 'Top 3 Choices', take 1st choice.
2. Data Cleaning:
   - Dropped missing rows & 'No Cat Experience' respondents.
   - Standardized breeds and brands from config.
3. Feature Encoding: Ordinal & One-Hot Encoding.
4. Feature Selection: ANOVA test + RandomForest (parallel execution).
5. Text Mining: Boolean features extracted from config insights.

=== Execution Summary ===
- Data Profiling: Parallel (4 profilers)
- Feature Selection: Parallel (ANOVA + RandomForest)
- Chart Generation: Parallel (7 charts)
- Total charts generated: 7
"""
    with open('presentation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    logging.info("Exported: presentation_summary.txt")
except Exception as e:
    logging.error(f"Failed to export summary text: {e}")

logging.info(f"[Diagnostic] Export completed in {time.time() - t_export:.2f}s")
logging.info("\n[OK] Pipeline completed successfully!")
