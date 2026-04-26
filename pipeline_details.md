# Pipeline Overview — รายละเอียดแต่ละขั้นตอน

---

## Pipeline Flow Diagram

```
Raw CSV (169 rows, 76 columns)
    │
    ▼
Step 1: Target Variable Identification
    │ → Extract 1st choice from "Top 3 Choices" column
    │ → 10 unique target classes (Option 1–10)
    ▼
Step 2: Data Cleaning
    │ → dropna(subset=[Target_Option, Age, Gender])  → -21 rows
    │ → Experience filter (เคย/ไม่เคย)
    │ → Breed grouping → Purebred / Mixed-Stray
    │ → Brand standardization → canonical brand names
    ▼
Step 3: Feature Selection & Engineering
    │ → Likert 5-level → 1-5  (13 columns)
    │ → Option Rating 4-level → 1-4  (50 columns)
    │ → Ordinal Encode Age  (1-4)
    │ → One-Hot Encode Gender + Marital Status
    │ → Text Mining → 9 Boolean insight features
    │ → ANOVA F-test → Top 10 Features by lowest p-value
    ▼
Step 4: Data Visualization
    │ → chart1: Demographics (age, gender, marital)
    │ → chart2: Target Distribution (Option 1–10 count)
    │ → chart3: Correlation Heatmap (Top 10 Features)
    │ → chart4: Mean Option Scores (1-10, 5 attributes)
    │ → chart5: Insights Distribution (9 boolean features)
    │ → chart6: Insights vs Target Option
    ▼
Cleaned CSV (148 rows, 90+ columns) + 6 PNG charts + presentation
```

---

## Step 1 — Target Variable Identification

### Source Column
`Top 3 Choices` — คอลัมน์ที่ 72 (index 72) ใน raw CSV

### Sample Values
```
"Option 3, Option 1, Option 5"
"Option 7"
"Option 2, Option 9, Option 4"
```

### Logic
```python
# แยกด้วย comma แล้วใช้ตัวแรก
choices = row['Top 3 Choices'].split(',')
target = choices[0].strip()  # e.g. "Option 3"
```

### Result
| Target_Option | Count | % |
|---|---|---|
| Option 3 🏆 | 36 | 24.3% |
| Option 1 | 23 | 15.5% |
| Option 2 | 22 | 14.9% |
| Option 6 | 19 | 12.8% |
| Option 7 | 11 | 7.4% |
| Option 4 | 5 | 3.4% |
| Option 5 | 5 | 3.4% |
| Option 8 | 5 | 3.4% |
| Option 9 | 5 | 3.4% |
| Option 10 | 5 | 3.4% |

---

## Step 2 — Data Cleaning

### 2.1 Drop rows with NaN in key columns

**Key Columns:** `Target_Option`, `Age`, `Gender`

```python
important_cols = ['Target_Option', 'Age', 'Gender']
df_clean = df.dropna(subset=important_cols)
# 169 → 148 rows (−21 rows)
```

### 2.2 Experience Filter

```python
df_clean = df_clean[df_clean['Experience'] == 'เคย']
# ทุก row ที่เหลือมี Experience == 'เคย' อยู่แล้ว → 0 rows removed
```

### 2.3 Breed Grouping

```python
def categorize_breed(breed_str):
    purebred_keywords = [
        'british', 'persian', 'scottish', 'munchkin', 'ragdoll',
        'siamese', 'maine coon', 'bengal', 'sphynx', 'himalayan',
        'abyssinian', 'russian', 'thai', 'วิเชียรมาศ', 'หิมาลายัน',
        'ขาวมณี', 'โกนจา', 'ศุภลักษณ์', 'สีสวาด', 'โคราช'
    ]
    # ถ้าพบ keyword ใดใน breed_str → Purebred
    # ไม่พบ → Mixed/Stray
```

### 2.4 Brand Standardization

```python
brand_mapping = {
    'whiskas': ['whiskas', 'whiskā', 'วิสกัส', 'วิสกัต', 'วิสคัส'],
    'royal canin': ['royal canin', 'รอยัลคานิน'],
    'smartheart': ['smartheart', 'smarth'],
    'sheba': ['sheba'],
    'me-o': ['me-o'],
    'projen': ['projen'],
    'king cat': ['king cat'],
    'you-o': ['you-o'],
    'kativa': ['kativa', 'kaniva'],
    'pramy': ['pramy'],
    'wills': ['wills'],
    'purina': ['purina'],
    "hill's": ["hill's"],
    'buzz': ['buzz'],
    'solid gold': ['solid gold'],
    'taste of the wild': ['taste of the wild'],
    'perfecta': ['perfecta'],
    'maxima': ['maxima'],
    'hero cat': ['hero cat'],
    'oliver': ['oliver']
}
```

---

## Step 3 — Feature Selection & Engineering

### 3.1 Likert 5-level → 1-5 (13 columns)

```python
likert_map = {
    'มากที่สุด': 5,
    'มาก':      4,
    'ปานกลาง':  3,
    'น้อย':     2,
    'น้อยที่สุด': 1
}
```

**Applied to:**
- `Factor_Taste`, `Factor_Brand_Rep`, `Factor_Price`, `Factor_Ingredients`, `Factor_Packaging` (5 cols)
- `PkgFactor_Taste`, `PkgFactor_Freshness`, `PkgFactor_Nutrition`, `PkgFactor_Safety`, `PkgFactor_Convenience`, `PkgFactor_Promotion`, `PkgFactor_Quality`, `PkgFactor_Price` (8 cols)

### 3.2 Option Rating 4-level → 1-4 (50 columns)

```python
rating_map = {
    'เห็นด้วยที่สุด': 4,
    'เห็นด้วยอย่างยิ่ง': 4,
    'เห็นด้วย': 3,
    'เฉยๆ': 2,
    'ไม่เห็นด้วย': 1,
    'ไม่เห็นด้วยเลย': 1
}
```

**Applied to:** 10 Options × 5 Attributes = 50 columns
- Attributes: `Attractiveness`, `Trust`, `Modernity`, `Premium_Feel`, `Purchase_Intent`

### 3.3 Ordinal Encoding — Age

```python
age_map = {
    '20-29 ปี': 1,
    '30-39 ปี': 2,
    '40-49 ปี': 3,
    '50+ ปี':   4
}
```

### 3.4 One-Hot Encoding — Gender + Marital Status

```python
# Gender → 3 columns
Gender_Male   = 1 if row['Gender'] == 'ชาย' else 0
Gender_Female = 1 if row['Gender'] == 'หญิง' else 0
Gender_Other  = 1 if row['Gender'] == 'อื่นๆ' else 0

# Marital Status → 4 columns
Marital_Single         = 1 if row['Marital'] == 'โสด' else 0
Marital_Married        = 1 if row['Marital'] == 'แต่งงานแล้ว' else 0
Marital_In_Relationship = 1 if row['Marital'] == 'มีแฟนแต่ยังไม่แต่งงาน' else 0
Marital_Divorced       = 1 if row['Marital'] == 'หย่าร้าง/เป็นม่าย' else 0
```

### 3.5 Text Mining — 9 Boolean Insight Features

```python
def extract_insights(row):
    text = str(row.get('Packaging_Suggestion', '')) + ' ' + str(row.get('Current_Brand_Detail', ''))

    insights = {
        'Need_Clear_Window': any(k in text for k in ['ใส', 'มองเห็น', 'ทะลุ']),
        'Need_Pour_Lid':      any(k in text for k in ['ฝา', 'ขวด', 'เท']),
        'Need_Small_Packs':   any(k in text for k in ['แบ่ง', 'ถุงเล็ก', 'ห่อย่อย', 'ซองย่อย', 'เล็กๆ']),
        'Need_Sodium_Info':   any(k in text for k in ['โซเดียม', 'เค็ม', 'ไต']),
        'Need_Big_Text':      any(k in text for k in ['ตัวใหญ่', 'ชัดเจน', 'อ่านง่าย']),
        'Design_Matte':       any(k in text for k in ['ด้าน', 'เนื้อด้าน']),
        'Design_Cartoon':     any(k in text for k in ['การ์ตูน', 'วาด', 'ตาโต']),
        'Need_Topping':       any(k in text for k in ['ท็อปปิ้ง', 'เนื้อจริง', 'ฟรีซดราย', 'ผสมเนื้อ']),
        'Need_Interactive':   any(k in text for k in ['ของแถม', 'กล่องสุ่ม', 'ดม', 'ลับเล็บ', 'เล่น'])
    }
    return insights
```

### 3.6 ANOVA Feature Selection

```python
from scipy.stats import f_oneway

# สำหรับแต่ละ Feature ที่ encode แล้ว:
groups = [df[df['Target_Option'] == opt][feature] for opt in range(1, 11)]
f_stat, p_value = f_oneway(*groups)

# เลือก 10 features ที่มี p-value ต่ำที่สุด
top10 = features.nsmallest(10, 'p_value')
```

---

## Step 4 — Data Visualization

### chart1: Demographics
- กราฟแท่ง/วงกลม แสดงอายุ, เพศ, สถานะครอบครัวของผู้ตอบ 148 คน
- กลุ่มอายุ 30–39 ปี มากที่สุด (62 คน, 41.9%)

### chart2: Target Distribution
- กราฟแท่งแสดงจำนวนเสียงของแต่ละ Option (1–10)
- Option 3 นำโด่งที่ 36 เสียง (24.3%)

### chart3: Correlation Heatmap
- แสดง correlation ระหว่าง Top 10 Features
- Features ภายในกลุ่มเดียวกัน (เช่น Attractiveness ของ Option ต่างๆ) มี correlation สูง

### chart4: Mean Option Scores
- คะแนนเฉลี่ยของแต่ละ Option ใน 5 มิติ (1-4 scale)
- Option 3 มีคะแนนเฉลี่ยสูงสุด (2.56)

### chart5: Insights Distribution
- กราฟแท่งแสดงจำนวนครั้งที่แต่ละ Insight Feature ถูกกล่าวถึง
- Need_Small_Packs สูงสุด (25 ครั้ง, 17%)

### chart6: Insights vs Target Option
- แสดงว่าผู้ที่เลือกแต่ละ Option มี insight features อะไรบ้าง
- Option 3 โดดเด่นในทุก insight categories

---

## Output Files

| ไฟล์ | รายละเอียด |
|------|-----------|
| `cleaned_survey_data.csv` | ข้อมูลสะอาด 148 rows × 90+ columns |
| `chart1_demographics.png` | กราฟประชากรศาสตร์ |
| `chart2_target_distribution.png` | กราฟการกระจายตัวเลือก |
| `chart3_correlation_heatmap.png` | Correlation Heatmap |
| `chart4_mean_option_scores.png` | คะแนนเฉลี่ย Option |
| `chart5_insights_distribution.png` | กระจายตัว Insights |
| `chart6_insights_vs_option.png` | Insights vs Option |
| `presentation_AIE323.md` | Slide deck |
| `presentation_summary.txt` | สรุป methodology |
