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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG: ตั้งค่า Font ภาษาไทย (ปรับ path ตามเครื่อง)
# ============================================================
# สำหรับ Google Colab ให้ใช้: 
#   !apt-get install -y fonts-tlwg-loma
#   matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/tlwg/Loma.ttf')
#   plt.rcParams['font.family'] = 'Loma'
# สำหรับ Windows ที่มี Tahoma:
try:
    plt.rcParams['font.family'] = 'Tahoma'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA...")
print("=" * 60)

CSV_FILE = 'BU Data from Survey Cases_final(5).csv'
df_raw = pd.read_csv(CSV_FILE, encoding='cp874', header=1)
print(f"Raw shape: {df_raw.shape}")

# ============================================================
# RENAME COLUMNS (ชื่อคอลัมน์ภาษาไทยถูก corrupt เป็น ?)
# ============================================================
col_names = {
    0: 'Timestamp',
    1: 'Experience',           # คุณเคยซื้ออาหารแมว...
    2: 'Cat_Meaning',          # แมวมีความหมายอย่างไร
    3: 'Cat_Breed',            # พันธุ์แมว
    4: 'Current_Brand',        # แบรนด์ที่ซื้อปัจจุบัน
    5: 'Factor_Taste',         # ปัจจัย: รสชาติ
    6: 'Factor_Brand_Rep',     # ปัจจัย: ชื่อเสียงแบรนด์
    7: 'Factor_Price',         # ปัจจัย: ราคา/โปรโมชั่น
    8: 'Factor_Ingredients',   # ปัจจัย: ส่วนผสม/สารอาหาร
    9: 'Factor_Packaging',     # ปัจจัย: บรรจุภัณฑ์
    10: 'Packaging_Important', # packaging สำคัญไหม
    11: 'Design_Preference',   # ชอบแบบไหน (ถ่ายจริง/AI/วาด)
    12: 'PkgFactor_Taste',     # ปัจจัยบรรจุภัณฑ์: รสชาติ
    13: 'PkgFactor_Freshness', # ปัจจัย: ความสดใหม่
    14: 'PkgFactor_Nutrition',  # ปัจจัย: สารอาหาร
    15: 'PkgFactor_Safety',    # ปัจจัย: ความปลอดภัย
    16: 'PkgFactor_Convenience', # ปัจจัย: ความสะดวก
    17: 'PkgFactor_Promotion',  # ปัจจัย: โปรโมชั่น
    18: 'PkgFactor_Quality',    # ปัจจัย: คุณภาพ
    19: 'PkgFactor_Price',      # ปัจจัย: ราคา
    20: 'Current_Brand_Detail', # แบรนด์ที่ชอบ (text)
    21: 'Packaging_Suggestion', # ข้อเสนอแนะบรรจุภัณฑ์
}

# Option 1-10, each has 5 sub-attributes
option_attrs = ['Attractiveness', 'Trust', 'Modernity', 'Premium_Feel', 'Purchase_Intent']
for opt in range(1, 11):
    for j, attr in enumerate(option_attrs):
        col_idx = 22 + (opt - 1) * 5 + j
        col_names[col_idx] = f'Opt{opt}_{attr}'

col_names[72] = 'Top3_Choices'
col_names[73] = 'Age'
col_names[74] = 'Gender'
col_names[75] = 'Marital_Status'

df = df_raw.copy()
df.columns = [col_names.get(i, f'col_{i}') for i in range(len(df.columns))]
print(f"Columns renamed: {len(df.columns)} columns")

# ============================================================
# STEP 1: TARGET VARIABLE IDENTIFICATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: TARGET VARIABLE IDENTIFICATION")
print("=" * 60)

df['Target_Option'] = df['Top3_Choices'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) else np.nan
)
print(f"Target_Option created. Distribution:")
print(df['Target_Option'].value_counts())

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# 2a. Drop rows that are completely NaN (except Timestamp/Experience)
important_cols = ['Target_Option', 'Age', 'Gender']
df_clean = df.dropna(subset=important_cols)
print(f"After dropping NaN in key columns: {len(df_clean)} rows (from {len(df)})")

# 2b. Handle Experience filter (Col 1)
# len=3 → เคย (Yes), len=6 → ไม่เคย (No)
exp_map = {}
for val in df_clean['Experience'].dropna().unique():
    if len(str(val)) == 3:
        exp_map[val] = 'Yes'
    elif len(str(val)) == 6:
        exp_map[val] = 'No'
    else:
        exp_map[val] = 'Unknown'
df_clean['Experience'] = df_clean['Experience'].map(exp_map)

# Logic Check: Flag rows where Experience='No' but they filled detailed ratings
no_exp = df_clean[df_clean['Experience'] == 'No']
print(f"Respondents with no experience: {len(no_exp)}")
# Keep them flagged but don't remove — they may still have valid packaging opinions
df_clean['Has_Experience'] = (df_clean['Experience'] == 'Yes').astype(int)

# 2c. Standardize Cat Breed (Col 3)
def categorize_breed(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    purebreds = ['british', 'persian', 'scottish', 'munchkin', 'ragdoll',
                 'siamese', 'exotic', 'maine', 'bengal', 'american',
                 'sphynx', 'himalayan', 'abyssinian', 'russian']
    for breed in purebreds:
        if breed in text:
            return 'Purebred'
    return 'Mixed/Stray'

df_clean['Cat_Breed_Cat'] = df_clean['Cat_Breed'].apply(categorize_breed)
print(f"Cat Breed categories: {df_clean['Cat_Breed_Cat'].value_counts().to_dict()}")

# 2d. Standardize Current Brand (Col 4)
def standardize_brand(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    brand_map = {
        'royal canin': 'Royal Canin', 'royalcanin': 'Royal Canin',
        'whiskas': 'Whiskas', 'whiska': 'Whiskas',
        'me-o': 'Me-O', 'meo': 'Me-O',
        'projen': 'Projen',
        'king cat': 'King Cat', 'kingcat': 'King Cat',
        'you-o': 'You-O', 'you o': 'You-O',
        'sheba': 'Sheba',
        'catival': 'Catival', 'katival': 'Catival',
        'pramy': 'Pramy',
        'wills': 'Wills', 'will': 'Wills',
        'smarth': 'SmartHeart', 'smart heart': 'SmartHeart',
        'purina': 'Purina', 'friskies': 'Purina',
        'hills': "Hill's", "hill's": "Hill's",
    }
    for keyword, std_name in brand_map.items():
        if keyword in text:
            return std_name
    return 'Other'

df_clean['Brand_Std'] = df_clean['Current_Brand'].apply(standardize_brand)
print(f"Brand categories: {df_clean['Brand_Std'].value_counts().head(10).to_dict()}")

print(f"\nCleaned dataset shape: {df_clean.shape}")

# ============================================================
# STEP 3: FEATURE SELECTION & ENCODING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE SELECTION & ENCODING")
print("=" * 60)

# 3a. Likert Scale → Numeric (Purchase Decision Factors, cols 5-9)
# len=3→มาก=4, len=4→น้อย=2, len=7→ปานกลาง=3, len=9→มากที่สุด=5, len=10→น้อยที่สุด=1
def map_likert_5(val):
    if pd.isna(val):
        return np.nan
    l = len(str(val))
    return {9: 5, 3: 4, 7: 3, 4: 2, 10: 1}.get(l, np.nan)

likert5_cols = [
    'Factor_Taste', 'Factor_Brand_Rep', 'Factor_Price',
    'Factor_Ingredients', 'Factor_Packaging',
    'PkgFactor_Taste', 'PkgFactor_Freshness', 'PkgFactor_Nutrition',
    'PkgFactor_Safety', 'PkgFactor_Convenience', 'PkgFactor_Promotion',
    'PkgFactor_Quality', 'PkgFactor_Price'
]
for col in likert5_cols:
    df_clean[col] = df_clean[col].apply(map_likert_5)
print(f"Likert-5 encoded: {len(likert5_cols)} columns")

# 3b. Option Ratings → Numeric (4-point scale)
# len=14→เห็นด้วยที่สุด=4, len=8→เห็นด้วย=3, len=4→เฉยๆ=2, len=11→ไม่เห็นด้วย=1
def map_likert_4(val):
    if pd.isna(val):
        return np.nan
    l = len(str(val))
    return {14: 4, 8: 3, 4: 2, 11: 1}.get(l, np.nan)

option_cols = [c for c in df_clean.columns if c.startswith('Opt')]
for col in option_cols:
    df_clean[col] = df_clean[col].apply(map_likert_4)
print(f"Option ratings encoded: {len(option_cols)} columns")

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
    else:  # 50+
        age_map[val] = 4
df_clean['Age_Ordinal'] = df_clean['Age'].map(age_map)
print(f"Age ordinal encoded: {age_map}")

# 3d. One-Hot Encoding for Gender
gender_map = {}
for val in df_clean['Gender'].dropna().unique():
    l = len(str(val))
    if l == 3:
        gender_map[val] = 'Male'
    elif l == 4:
        gender_map[val] = 'Female'
    else:
        gender_map[val] = 'Other'
df_clean['Gender_Label'] = df_clean['Gender'].map(gender_map)
gender_dummies = pd.get_dummies(df_clean['Gender_Label'], prefix='Gender')
df_clean = pd.concat([df_clean, gender_dummies], axis=1)

# 3e. One-Hot Encoding for Marital Status
marital_map = {}
for val in df_clean['Marital_Status'].dropna().unique():
    l = len(str(val))
    if l == 11:
        marital_map[val] = 'Single'
    elif l == 21:
        marital_map[val] = 'Married_Children'
    elif l == 12:
        marital_map[val] = 'Married_NoChildren'
    else:
        marital_map[val] = 'Divorced'
df_clean['Marital_Label'] = df_clean['Marital_Status'].map(marital_map)
marital_dummies = pd.get_dummies(df_clean['Marital_Label'], prefix='Marital')
df_clean = pd.concat([df_clean, marital_dummies], axis=1)
print(f"One-Hot encoded Gender & Marital Status")

# 3f. Feature Selection using Chi-Square / ANOVA
print("\n--- Feature Selection: Top 10 features ---")

# Encode Target for analysis
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target_encoded = le.fit_transform(df_clean['Target_Option'].dropna())

# Collect all numeric feature columns
feature_cols = likert5_cols + option_cols + ['Age_Ordinal', 'Has_Experience']
feature_cols += [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]

# ANOVA for numeric features
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
        group_data = df_clean.loc[
            valid_mask & (df_clean['Target_Option'] == opt), col
        ].dropna()
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
print(top10.to_string())

top10_features = top10.index.tolist()
print(f"\nTop 10 features: {top10_features}")

# ============================================================
# STEP 4: DATA VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: DATA VISUALIZATION")
print("=" * 60)

# Color palette
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

fig_num = 1

# 4a. Demographic Profile
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Demographic Profile of Respondents', fontsize=16, fontweight='bold')

# Age
age_counts = df_clean['Age_Ordinal'].value_counts().sort_index()
age_labels = ['20-29', '30-39', '40-49', '50+']
axes[0].bar(age_labels[:len(age_counts)], age_counts.values, color=COLORS[:len(age_counts)], edgecolor='white')
axes[0].set_title('Age Distribution', fontsize=13)
axes[0].set_ylabel('Count')
for i, v in enumerate(age_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Gender
gender_counts = df_clean['Gender_Label'].value_counts()
axes[1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
            colors=COLORS[:len(gender_counts)], startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Gender Distribution', fontsize=13)

# Marital Status
marital_counts = df_clean['Marital_Label'].value_counts()
axes[2].barh(marital_counts.index, marital_counts.values, color=COLORS[:len(marital_counts)], edgecolor='white')
axes[2].set_title('Marital Status', fontsize=13)
for i, v in enumerate(marital_counts.values):
    axes[2].text(v + 0.3, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('chart1_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart {fig_num} saved: chart1_demographics.png")
fig_num += 1

# 4b. Target Variable Distribution
fig, ax = plt.subplots(figsize=(12, 6))
target_counts = df_clean['Target_Option'].value_counts().sort_index()
bars = ax.bar(target_counts.index, target_counts.values, color=COLORS[:len(target_counts)], edgecolor='white', linewidth=1.5)
ax.set_title('Distribution of Top 1 Packaging Choice (Target Variable)', fontsize=14, fontweight='bold')
ax.set_xlabel('Packaging Option')
ax.set_ylabel('Number of Votes')
for bar, val in zip(bars, target_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', fontweight='bold', fontsize=11)
ax.set_ylim(0, target_counts.max() * 1.15)
plt.tight_layout()
plt.savefig('chart2_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart {fig_num} saved: chart2_target_distribution.png")
fig_num += 1

# 4c. Correlation Heatmap (Top features vs Options)
# Calculate mean scores per Option for top features
opt_score_cols = [c for c in option_cols if any(c in top10_features for c in [c])]
heatmap_cols = [c for c in top10_features if c in df_clean.columns][:10]

if len(heatmap_cols) >= 2:
    corr_data = df_clean[heatmap_cols].dropna()
    if len(corr_data) > 5:
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                    fmt='.2f', ax=ax, linewidths=0.5, square=True)
        ax.set_title('Correlation Heatmap: Top 10 Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('chart3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart {fig_num} saved: chart3_correlation_heatmap.png")
        fig_num += 1

# 4d. BONUS: Mean Option Scores comparison
opt_means = {}
for opt in range(1, 11):
    opt_cols_i = [c for c in df_clean.columns if c.startswith(f'Opt{opt}_')]
    if opt_cols_i:
        opt_means[f'Option {opt}'] = df_clean[opt_cols_i].mean().mean()

if opt_means:
    fig, ax = plt.subplots(figsize=(12, 6))
    opts = list(opt_means.keys())
    vals = list(opt_means.values())
    bars = ax.bar(opts, vals, color=COLORS[:len(opts)], edgecolor='white', linewidth=1.5)
    ax.set_title('Mean Rating Score by Packaging Option', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Score (1-4)')
    ax.set_ylim(0, 4.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig('chart4_mean_option_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart {fig_num} saved: chart4_mean_option_scores.png")

# ============================================================
# STEP 5: EXPORT & SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: EXPORT & SUMMARY")
print("=" * 60)

# 5a. Select final columns for export
export_cols = (
    ['Target_Option', 'Has_Experience', 'Cat_Breed_Cat', 'Brand_Std']
    + likert5_cols + option_cols
    + ['Age_Ordinal', 'Gender_Label', 'Marital_Label']
    + [c for c in df_clean.columns if c.startswith('Gender_') or c.startswith('Marital_')]
)
export_cols = [c for c in export_cols if c in df_clean.columns]
df_export = df_clean[export_cols].copy()
df_export.to_csv('cleaned_survey_data.csv', index=False, encoding='utf-8-sig')
print(f"Exported: cleaned_survey_data.csv ({df_export.shape})")

# 5b. Summary for Presentation
print("\n" + "=" * 60)
print("SUMMARY FOR PRESENTATION SLIDES")
print("=" * 60)
# Write summary to file instead of console (to avoid Thai encoding issues)
summary_text = """
=== Logic for Target & Feature Selection ===

1. Target Variable: Extract column 'Top 3 Choices', take only
   the 1st choice (first item before comma) as Target_Option

2. Data Cleaning:
   - Filtered out 851 empty rows, keeping usable data
   - Grouped cat breeds into 'Purebred' vs 'Mixed/Stray'
   - Standardized brand names into consistent groups
   - Logic Check: Flagged respondents with no cat experience

3. Feature Encoding:
   - Likert Scale 5 levels -> numeric 1-5
   - Option Rating 4 levels -> numeric 1-4
   - Age -> Ordinal Encoding (1-4)
   - Gender/Marital -> One-Hot Encoding

4. Feature Selection:
   - Used ANOVA test to compare feature means across Target groups
   - Selected Top 10 Features with lowest p-values

=== Data Quality & Cleanliness ===

1. Removed rows with incomplete data (NaN in key columns)
2. Logic Check between cat experience and detailed responses
3. Standardized open-ended data (cat breeds, brands) into categories
4. Consistently encoded Likert Scales to numeric across all columns
5. Applied appropriate encoding by data type (Ordinal vs One-Hot)
"""
with open('presentation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(summary_text)

print("\n[OK] Pipeline completed successfully!")
print("   - Cleaned data: cleaned_survey_data.csv")
print("   - Summary: presentation_summary.txt")
print("   - Charts: chart1_demographics.png, chart2_target_distribution.png,")
print("             chart3_correlation_heatmap.png, chart4_mean_option_scores.png")
