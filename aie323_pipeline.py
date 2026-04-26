"""
AIE323 - Data Preparation Pipeline
===================================
โปรเจกต์วิเคราะห์ข้อมูลแบบสอบถาม (Survey Data) 
เรื่องการออกแบบบรรจุภัณฑ์อาหารแมวสำหรับแบรนด์ต่างประเทศที่ต้องการบุกตลาดไทย

ครอบคลุม 5 ขั้นตอน:
1. Target Variable Identification
2. Data Cleaning
3. Feature Selection & Encoding
4. Data Visualization
5. Export & Summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import warnings
import json
import logging
import sys

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
# RENAME COLUMNS (ชื่อคอลัมน์ภาษาไทยถูก corrupt เป็น ?)
# ============================================================
# Convert string keys from JSON to integers
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
# STEP 1: TARGET VARIABLE IDENTIFICATION
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STEP 1: TARGET VARIABLE IDENTIFICATION")
logging.info("=" * 60)

df['Target_Option'] = df['Top3_Choices'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) else np.nan
)
logging.info(f"Target_Option created. Distribution:\n{df['Target_Option'].value_counts().to_string()}")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STEP 2: DATA CLEANING")
logging.info("=" * 60)

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

logging.info(f"\nCleaned dataset shape: {df_clean.shape}")

# ============================================================
# STEP 3: FEATURE SELECTION & ENCODING
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STEP 3: FEATURE SELECTION & ENCODING")
logging.info("=" * 60)

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

# 3f. Feature Selection
logging.info("\n--- Feature Selection: Top 10 features ---")
feature_cols = likert5_cols + option_cols + ['Age_Ordinal']
feature_cols += [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]

scores = {}
valid_mask = df_clean['Target_Option'].notna()
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
top10 = scores_df.head(10)
logging.info(f"\n{top10.to_string()}")

top10_features = top10.index.tolist()
logging.info(f"\nTop 10 features: {top10_features}")

# 3g. Random Forest Feature Importance
logging.info("\n--- Feature Importance (Random Forest) ---")
try:
    # เตรียมข้อมูลสำหรับ Random Forest
    rf_data = df_clean.dropna(subset=['Target_Option']).copy()
    
    # เลือก Features ที่เป็นตัวเลขแล้ว
    exclude_cols = ['Gender_Label', 'Marital_Label', 'Marital_Status']
    rf_features = likert5_cols + option_cols + ['Age_Ordinal'] + [c for c in df_clean.columns if (c.startswith('Gender_') or c.startswith('Marital_')) and c not in exclude_cols]
    rf_features = [c for c in rf_features if c in rf_data.columns]
    
    X = rf_data[rf_features].fillna(0) # เติม 0 แทน NaN เพื่อให้รันได้
    
    le = LabelEncoder()
    y = le.fit_transform(rf_data['Target_Option'])
    
    # 1. ใช้ SelectKBest กรอง Feature ก่อน (เลือกมา 20 ตัวที่สัมพันธ์กับ Target ที่สุด)
    k_best = min(20, len(rf_features))
    selector = SelectKBest(score_func=chi2, k=k_best)
    X_kbest = selector.fit_transform(X, y)
    
    # 2. ดึงชื่อ Feature ที่ผ่านการคัดเลือก
    selected_mask = selector.get_support()
    selected_features = [rf_features[i] for i in range(len(rf_features)) if selected_mask[i]]
    logging.info(f"Filtered {len(rf_features)} features down to {k_best} using SelectKBest(chi2).")
    
    # 3. นำ Feature ที่คัดแล้วไปเข้า Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_kbest, y)
    
    rf_scores = pd.DataFrame({'Feature': selected_features, 'Importance': rf.feature_importances_})
    rf_scores = rf_scores.sort_values('Importance', ascending=False)
    
    logging.info(f"\nTop 10 Features (Random Forest):\n{rf_scores.head(10).to_string(index=False)}")
except Exception as e:
    logging.warning(f"Failed to calculate Random Forest importance: {e}")

# 3h. Text Mining & Insight Extraction
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
# STEP 4: DATA VISUALIZATION
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STEP 4: DATA VISUALIZATION")
logging.info("=" * 60)

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
fig_num = 1

def save_plot(filename):
    global fig_num
    try:
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Chart {fig_num} saved: {filename}")
        fig_num += 1
    except Exception as e:
        logging.warning(f"Failed to save chart {filename}: {e}")
        plt.close()

# 4a. Demographic Profile
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
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
    
    save_plot('chart1_demographics.png')
except Exception as e:
    logging.warning(f"Error in generating demographic plot: {e}")

# 4b. Target Variable Distribution
try:
    fig, ax = plt.subplots(figsize=(12, 6))
    target_counts = df_clean['Target_Option'].value_counts().sort_index()
    ax.bar(target_counts.index, target_counts.values, color=COLORS[:len(target_counts)], edgecolor='white')
    ax.set_title('Distribution of Top 1 Packaging Choice (Target Variable)', fontsize=14, fontweight='bold')
    save_plot('chart2_target_distribution.png')
except Exception as e:
    logging.warning(f"Error in generating target distribution plot: {e}")

# 4c. Correlation Heatmap
try:
    heatmap_cols = [c for c in top10_features if c in df_clean.columns][:10]
    if len(heatmap_cols) >= 2:
        corr_data = df_clean[heatmap_cols].dropna()
        if len(corr_data) > 5:
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = corr_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax, linewidths=0.5, square=True)
            ax.set_title('Correlation Heatmap: Top 10 Features', fontsize=14, fontweight='bold')
            save_plot('chart3_correlation_heatmap.png')
except Exception as e:
    logging.warning(f"Error in generating correlation heatmap: {e}")

# 4d. Mean Option Scores
try:
    opt_means = {}
    for opt in range(1, 11):
        opt_cols_i = [c for c in df_clean.columns if c.startswith(f'Opt{opt}_')]
        if opt_cols_i:
            opt_means[f'Option {opt}'] = df_clean[opt_cols_i].mean().mean()

    if opt_means:
        fig, ax = plt.subplots(figsize=(12, 6))
        opts = list(opt_means.keys())
        vals = list(opt_means.values())
        ax.bar(opts, vals, color=COLORS[:len(opts)], edgecolor='white')
        ax.set_title('Mean Rating Score by Packaging Option', fontsize=14, fontweight='bold')
        save_plot('chart4_mean_option_scores.png')
except Exception as e:
    logging.warning(f"Error in generating mean scores plot: {e}")

# 4e. Qualitative Insights Distribution
try:
    if insight_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        insight_counts = df_clean[insight_cols].sum().sort_values(ascending=True)
        ax.barh(insight_counts.index, insight_counts.values, color=COLORS[0], edgecolor='white')
        ax.set_title('Top Requested Features & Pain Points (from Text)', fontsize=14, fontweight='bold')
        save_plot('chart5_insights_distribution.png')
except Exception as e:
    logging.warning(f"Error in generating insights distribution plot: {e}")

# 4f. Top Insights vs Target Option
try:
    if insight_cols:
        top_insights = df_clean[insight_cols].sum().sort_values(ascending=True).tail(5).index.tolist()
        plot_data = pd.DataFrame()
        for col in top_insights:
            counts = df_clean[df_clean[col] == 1].groupby('Target_Option').size()
            plot_data[col] = counts
        plot_data = plot_data.fillna(0)
        
        if not plot_data.empty:
            ax = plot_data.T.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set3')
            ax.set_title('Top 5 Requested Features by Target Packaging Option', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            save_plot('chart6_insights_vs_option.png')
except Exception as e:
    logging.warning(f"Error in generating insights vs option plot: {e}")

# 4g. Random Forest Feature Importance Plot
try:
    if 'rf_scores' in locals() and not rf_scores.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        # ใช้ barplot จาก seaborn สำหรับแสดงผล
        sns.barplot(x='Importance', y='Feature', data=rf_scores.head(10), palette='viridis', ax=ax)
        ax.set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('Features')
        save_plot('chart7_rf_feature_importance.png')
except Exception as e:
    logging.warning(f"Error in generating RF importance plot: {e}")

# ============================================================
# STEP 5: EXPORT & SUMMARY
# ============================================================
logging.info("\n" + "=" * 60)
logging.info("STEP 5: EXPORT & SUMMARY")
logging.info("=" * 60)

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
    summary_text = """=== Logic for Target & Feature Selection ===
1. Target Variable: Extract 'Top 3 Choices', take 1st choice.
2. Data Cleaning:
   - Dropped missing rows & 'No Cat Experience' respondents.
   - Standardized breeds and brands from config.
3. Feature Encoding: Ordinal & One-Hot Encoding.
4. Feature Selection: ANOVA test for Top 10 features.
5. Text Mining: Boolean features extracted from config insights.
"""
    with open('presentation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    logging.info("Exported: presentation_summary.txt")
except Exception as e:
    logging.error(f"Failed to export summary text: {e}")

logging.info("\n[OK] Pipeline completed successfully!")
