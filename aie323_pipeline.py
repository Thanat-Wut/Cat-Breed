"""
โปรเจกต์วิเคราะห์ข้อมูลแบบสอบถาม (Survey Data)
เรื่องการออกแบบบรรจุภัณฑ์อาหารแมวสำหรับแบรนด์ต่างประเทศที่ต้องการบุกตลาดไทย

Sequential Pipeline:
- Stage 1: Target Variable Identification
- Stage 2: Data Cleaning
- Stage 3: Feature Selection & Encoding
- Stage 4: Data Visualization (7 charts, sequential)
- Stage 5: Export
"""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
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

try:
    plt.rcParams['font.family'] = 'Tahoma'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

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
    df_raw = pd.read_csv(CSV_FILE, encoding='utf-8-sig', header=1, nrows=167)
    logging.info(f"Raw shape: {df_raw.shape}")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    sys.exit(1)

col_names = {int(k): v for k, v in config['col_names'].items()}

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

important_cols = ['Target_Option', 'Age', 'Gender']
df_clean = df.dropna(subset=important_cols)
logging.info(f"After dropping NaN in key columns: {len(df_clean)} rows (from {len(df)})")

df_clean = df_clean[df_clean['Experience'] == 'เคย']
df_clean['Experience'] = 'Yes'

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

likert5_map = config['likert_5_map']
likert4_map = config['likert_4_map']

likert5_cols = [
    'Factor_Taste', 'Factor_Brand_Rep', 'Factor_Price',
    'Factor_Ingredients', 'Factor_Packaging',
    'PkgFactor_Taste', 'PkgFactor_Freshness', 'PkgFactor_Nutrition',
    'PkgFactor_Safety', 'PkgFactor_Convenience', 'PkgFactor_Promotion',
    'PkgFactor_Quality', 'PkgFactor_Price'
]
for col in likert5_cols:
    df_clean[col] = df_clean[col].map(likert5_map).fillna(df_clean[col])
logging.info(f"Likert-5 encoded: {len(likert5_cols)} columns")

option_cols = [c for c in df_clean.columns if c.startswith('Opt')]
for col in option_cols:
    df_clean[col] = df_clean[col].map(likert4_map).fillna(df_clean[col])
logging.info(f"Option ratings encoded: {len(option_cols)} columns")

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

df_clean['Gender_Label'] = df_clean['Gender'].map(config['gender_map']).fillna('Other')
gender_dummies = pd.get_dummies(df_clean['Gender_Label'], prefix='Gender')
df_clean = pd.concat([df_clean, gender_dummies], axis=1)

df_clean['Marital_Label'] = df_clean['Marital_Status'].map(config['marital_map']).fillna('Other')
marital_dummies = pd.get_dummies(df_clean['Marital_Label'], prefix='Marital')
df_clean = pd.concat([df_clean, marital_dummies], axis=1)
logging.info(f"One-Hot encoded Gender & Marital Status")

logging.info(f"[Diagnostic] Encoding completed in {time.time() - t_stage3:.2f}s")

# ============================================================
# STAGE 3B: ANOVA FEATURE SELECTION
# ============================================================
logging.info("\n--- Feature Selection: Running ANOVA ---")
t_fselect = time.time()

feature_cols = likert5_cols + option_cols + ['Age_Ordinal']
feature_cols += [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]
valid_mask = df_clean['Target_Option'].notna()

def run_feature_selection():
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
            except Exception:
                pass
    scores_df = pd.DataFrame(scores).T.sort_values('p_value')
    return scores_df.head(10)

anova_result = run_feature_selection()

top10_features = anova_result.index.tolist()
logging.info(f"\n[ANOVA] Top 10 Features:\n{anova_result.to_string()}")
logging.info(f"[Diagnostic] Feature selection completed in {time.time() - t_fselect:.2f}s")

# ============================================================
# STAGE 3C: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
logging.info("\n--- Feature Selection: Random Forest Importance ---")
t_rf = time.time()

rf_feature_cols = likert5_cols + option_cols + ['Age_Ordinal']
rf_feature_cols += [c for c in df_clean.columns if c.startswith('Gender_') and c not in ('Gender_Label', 'Gender')]
rf_feature_cols += [c for c in df_clean.columns if c.startswith('Marital_') and c not in ('Marital_Label', 'Marital_Status')]
rf_feature_cols = [c for c in rf_feature_cols if c in df_clean.columns]

X_rf = df_clean[rf_feature_cols].apply(pd.to_numeric, errors='coerce')
X_rf = X_rf.fillna(X_rf.median())
le = LabelEncoder()
y_rf = le.fit_transform(df_clean['Target_Option'].fillna('Unknown'))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_rf, y_rf)

rf_importance = pd.Series(rf_model.feature_importances_, index=rf_feature_cols).sort_values(ascending=False)
logging.info(f"\n[RF] Top 10 Feature Importance:\n{rf_importance.head(10).to_string()}")
logging.info(f"[Diagnostic] RF feature selection completed in {time.time() - t_rf:.2f}s")

# ============================================================
# STAGE 3D: TEXT MINING & INSIGHT EXTRACTION
# ============================================================
logging.info("\n--- Insight Extraction (Text Mining) ---")
insight_cols = []
insights_map = config['insights_map']

combined_text = df_clean['Packaging_Suggestion'].fillna('') + ' ' + df_clean['Current_Brand_Detail'].fillna('')
for col_name, keywords in insights_map.items():
    pattern = '|'.join(keywords)
    df_clean[col_name] = combined_text.str.contains(pattern, case=False, na=False).astype(int)
    insight_cols.append(col_name)

logging.info(f"Extracted {len(insight_cols)} insight features from open-ended questions.")

# ============================================================
# STAGE 4: DATA VISUALIZATION (SEQUENTIAL)
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 4: DATA VISUALIZATION")
logging.info("=" * 60)
t_viz = time.time()

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

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
    top_n = rf_importance.head(10)
    if len(top_n) >= 2:
        fig = Figure(figsize=(12, 6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.barh(top_n.index[::-1], top_n.values[::-1], color=COLORS[0], edgecolor='white')
        ax.set_title('Random Forest Feature Importance (Top 10)', fontsize=14, fontweight='bold')
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

# Run all charts sequentially
for gen in chart_generators:
    gen()

logging.info(f"[Diagnostic] All charts generated in {time.time() - t_viz:.2f}s")

# ============================================================
# STAGE 5: EXPORT & SUMMARY
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STAGE 5: EXPORT & SUMMARY")
logging.info("=" * 60)
t_export = time.time()

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

logging.info(f"[Diagnostic] Export completed in {time.time() - t_export:.2f}s")
logging.info("\n[OK] Pipeline completed successfully!")
