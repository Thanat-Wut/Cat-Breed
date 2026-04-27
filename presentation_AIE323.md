# การจัดการและวิเคราะห์ข้อมูลแบบสอบถามพฤติกรรมการเลือกบรรจุภัณฑ์อาหารแมว
# Survey Data Management & Analysis for Cat Food Packaging Preference

---

## Slide 1: Title (หน้าปก)

**การจัดการและวิเคราะห์ข้อมูลแบบสอบถามพฤติกรรมการเลือกบรรจุภัณฑ์อาหารแมว**

**Survey Data Management & Analysis for Cat Food Packaging Preference**

**AIE323 Self-learning**

| | |
|---|---|
| **วิชา** | AIE323 Self-learning |
| **หัวข้อ** | การจัดการและวิเคราะห์ข้อมูลแบบสอบถามพฤติกรรมการเลือกบรรจุภัณฑ์อาหารแมว |
| **ชื่อสมาชิกกลุ่ม** | [กรอกชื่อสมาชิก] |
| **ภาคเรียน** | [ภาคเรียน/ปีการศึกษา] |

---

## Slide 2: Introduction & Business Brief (ที่มาและวัตถุประสงค์)

### บริบทธุรกิจ (Business Context)

แบรนด์อาหารแมวต่างประเทศต้องการบุกตลาดไทย จำเป็นต้องปรับบรรจุภัณฑ์ให้ตรงกับความต้องการของผู้บริโภคไทย จึงต้องวิเคราะห์ว่า **แบบบรรจุภัณฑ์ (Option 1–10) แบบใด** ที่ได้รับความนิยมมากที่สุด

### เป้าหมายของโปรเจกต์ (Project Goal)

นำข้อมูลจากแบบสอบถาม (Raw Survey Data) มาผ่านกระบวนการจัดการข้อมูลอย่างเป็นระบบ ให้ได้ไฟล์ CSV สุดท้าย พร้อมสำหรับการนำไปเทรน Machine Learning Model

### ขั้นตอนการทำงาน (Pipeline Overview)

```
ชุดข้อมูลจริง (167 rows) → Pipeline อ่าน CSV ด้วย header=1
    │ (834 แถวว่างที่อยู่หลังแถว 167 ไม่ใช่ส่วนหนึ่งของข้อมูลจริง)
    ▼
Step 1: Target Variable Identification
    │ → เลือก Target จาก "Top 3 Choices"
    ▼
Step 2: Data Cleaning
    │ → dropna(subset=[Target_Option, Age, Gender]) → ลบ 19 แถวที่ข้อมูลไม่สมบูรณ์
    │ → Experience filter (เคย/ไม่เคย — ผู้ที่ไม่เคยเลี้ยงแมวไม่ได้ตอบ)
    │ → Breed grouping → Purebred / Mixed-Stray (จาก config.json purebred_keywords)
    │ → Brand standardization → canonical brand names (จาก config.json brand_map)
    ▼
Step 3: Feature Selection & Engineering
    │ → Likert 5-level → 1-5  (13 columns)
    │ → Option Rating 4-level → 1-4  (50 columns)
    │ → Ordinal Encode Age  (1-4)
    │ → One-Hot Encode Gender + Marital Status
    │ → Text Mining → 9 Boolean insight features (จาก config.json insights_map)
    │ → ANOVA F-test → คัด Top 10 Features ที่มี p-value ต่ำสุด
    │ → Random Forest Feature Importance → วัดอิทธิพลบน 70 features ทั้งหมด
    ▼
Step 4: Data Visualization
    │ → chart1: Demographics (age, gender, marital)
    │ → chart2: Target Distribution (Option 1–10 count)
    │ → chart3: Correlation Heatmap (Top 10 ANOVA Features)
    │ → chart4: Mean Option Scores (1–10, 5 attributes)
    │ → chart5: Insights Distribution (9 boolean features)
    │ → chart6: Insights vs Target Option
    ▼
Cleaned CSV (148 rows, 89 columns) + 6 PNG charts + presentation
```

### Dataset Summary

| Metric | Value |
|--------|-------|
| ข้อมูลจริงในไฟล์ CSV | 167 rows |
| หลัง dropna (key cols NaN) | 148 rows |
| Features (original) | 76 columns |
| Features (after encoding + export) | 89 columns |
| Target Classes | 10 Options (1–10) |
| Survey Respondents | เจ้าของแมวในประเทศไทย |

---

## Slide 3: Step 1 - Target Variable Identification (การกำหนดตัวแปรเป้าหมาย)

### เป้าหมายหลักที่เลือกศึกษา

ในแบบสอบถาม ผู้ตอบถูกถามว่า **"เลือกแบบบรรจุภัณฑ์ที่ชอบที่สุด 3 อันดับแรก"** โดยตอบในรูปแบบ comma-separated เช่น `Option 3, Option 1, Option 5`

### วิธีการสร้าง Target Variable

```
"Top 3 Choices" column:
  "Option 3, Option 1, Option 5"
           │
           ▼  [แยกด้วย comma และใช้เฉพาะตัวเลือกแรก]
           ▼
Target_Option = "Option 3"
```

**กฎเกณฑ์:**
- แยกค่าด้วยเครื่องหมาย comma (`,`)
- ใช้เฉพาะ **ตัวเลือกแรก (1st choice)** เป็น Target Variable
- ตัดช่องว่างส่วนเกิน (strip whitespace)

### ผลลัพธ์ของการ Labeling

| Target_Option | Count | % |
|--------------|-------|---|
| Option 3 🏆 | 55 | 37.2% |
| Option 1 | 38 | 25.7% |
| Option 2 | 36 | 24.3% |
| Option 6 | 10 | 6.8% |
| Option 7 | 3 | 2.0% |
| Option 5 | 3 | 2.0% |
| Option 4 | 2 | 1.4% |
| Option 8 | 1 | 0.7% |
| Option 9 | 0 | 0% |
| Option 10 | 0 | 0% |

> **Insight:** Option 3 เป็นตัวเลือกที่ได้รับความนิยมสูงสุด คิดเป็น **37.2%** ของผู้ตอบทั้งหมด รองลงมาคือ Option 1 และ Option 2 ตามลำดับ สังเกตว่า Options 9 และ 10 ไม่มีผู้เลือกเลย

### Data Labeling Criteria

เนื่องจาก Target Variable คือ **Categorical (10 classes)** จึงเหมาะสำหรับ:
- **Classification** (เมื่อใช้โมเดลทำนายว่าผู้ตอบจะเลือก Option ใด)
- หรือ **ANOVA** (เมื่อวิเคราะห์ว่าตัวแปรใดมีผลต่อการเลือกแต่ละ Option)

---

## Slide 4: Step 2 - Data Cleaning (การทำความสะอาดและจัดการข้อมูล)

### การจัดการแบบสอบถามที่ตอบไม่ครบ (Handle Incomplete Responses)

ในชุดข้อมูลจริง 167 rows มีแถวที่มีค่า NaN ในคอลัมน์สำคัญ ได้แก่ `Target_Option`, `Age`, `Gender`

| ขั้นตอน | จำนวนแถว |
|--------|---------|
| ข้อมูลจริงในไฟล์ CSV | 167 |
| ลบแถว NaN ใน key columns | -19 |
| **Clean Data** | **148** |

> **19 แถว** ถูกลบออกเนื่องจากข้อมูลไม่สมบูรณ์ โดยทุกแถวที่ถูกลบเป็นผู้ที่ระบุว่า **"ไม่เคย"** เลี้ยงแมว (เลยไม่ได้ตอบคำถามส่วนที่เหลือ)

### การตรวจสอบความสมเหตุสมผล (Logic Check)

- ตรวจว่าค่า Likert ทุกคอลัมน์อยู่ในช่วง 1–5
- ตรวจว่า Option Rating ทุกคอลัมน์อยู่ในช่วง 1–4
- ตรวจว่าค่า Age เป็น ordinal ที่ถูกต้อง (1–4)

### Standardization: การจัดหมวดหมู่คำตอบจากคำถามปลายเปิด

**1. การจัดกลุ่มพันธุ์แมว (Breed Standardization)**

| กลุ่ม | จำนวน | สายพันธุ์ (keywords จาก config.json) |
|------|-------|-------------------------------------|
| **Mixed/Stray** 🏆 | 87 | แมวพันธุ์ผสม, แมวจรจัด |
| **Purebred** | 61 | british, persian, scottish, munchkin, ragdoll, siamese, exotic, maine, bengal, american, sphynx, himalayan, abyssinian, russian, เปอร์เซีย, สก็อต, มัชกิ้น, แรคดอล, เมนคูน, เบงกอล, วิเชียรมาศ, หิมาลายัน, ขาวมณี, โกนจา, ศุภลักษณ์, สีสวาด, โคราช |

**2. การมาตรฐานชื่อแบรนด์ (Brand Standardization)** จาก config.json

| Brand Group | Variants (keys) |
|------------|----------------|
| King Cat 🏆 | king cat, kingcat, kingkat, king cot, คิงแคท, คิงส์แคท, คิงงแคท, คิง แคท |
| Purina | purina, purino, friskies, purena, พูริโน่, ภูริโน, เพียวริน่า, ฟริสกี้ |
| You-O | you-o, you o, youo, ยูโอ |
| Kativa/Kaniva | catival, katival, kativa, cativa, kaniva, แคทิวา, แคทิว่า |
| Whiskas | whiskas, whiska, วิสกัส, วิสกัต, วิสคัส |
| Royal Canin | royal canin, royalcanin, รอยัลคานิน |
| SmartHeart | smarth, smart heart, สมาร์ทฮาร์ท, smartbrain |
| Me-O | me-o, meo, มีโอ |
| Buzz | buzz, บัซ |
| Others | Sheba, Pramy, Wills, Hill's, Solid Gold, Taste of the Wild, Petheria, Perfecta, Maxima, Hero Cat, Oliver |

**3. การสกัด Insight จากข้อความปลายเปิด**

นำคำตอบจาก `Packaging_Suggestion` และ `Current_Brand_Detail` มาสกัดเป็น **Boolean Features 9 ตัว** (จาก config.json insights_map):

| Feature | Keywords | Count | % |
|---------|----------|-------|---|
| Need_Big_Text | ตัวใหญ่, ชัดเจน, อ่านง่าย | 11 | 7.4% |
| Need_Interactive | ของแถม, กล่องสุ่ม, ดม, ลับเล็บ, เล่น | 10 | 6.8% |
| Need_Clear_Window | ใส, มองเห็น, ทะลุ | 8 | 5.4% |
| Need_Sodium_Info | โซเดียม, เค็ม, ไต | 7 | 4.7% |
| Need_Pour_Lid | ฝา, เท | 7 | 4.7% |
| Design_Cartoon | การ์ตูน, ตาโต | 7 | 4.7% |
| Design_Matte | ด้าน, เนื้อด้าน | 2 | 1.4% |
| Need_Small_Packs | แบ่ง, ถุงเล็ก, ซองย่อย | 1 | 0.7% |
| Need_Topping | ท็อปปิ้ง, เนื้อจริง | 0 | 0% |
| **(ไม่มี insight)** | — | 102 | 68.9% |

> รวมผู้ที่มี insightอย่างน้อย 1 อย่าง = **46 คน** จาก 148 (Design_Cartoon และ Need_Interactive มีค่าสูงสุด ตามด้วย Need_Sodium_Info และ Need_Pour_Lid)

---

## Slide 5: Step 3 - Feature Selection & Engineering (การแปลงและการคัดเลือกตัวแปร)

### Encoding: การแปลงข้อมูลเชิงคุณภาพให้เป็นตัวเลข

**1. Ordinal Encoding — อายุ (Age)**

| ช่วงอายุ | Ordinal Value |
|---------|--------------|
| 20–29 ปี | 1 |
| 30–39 ปี | 2 |
| 40–49 ปี | 3 |
| 50+ ปี | 4 |

**2. One-Hot Encoding — เพศและสถานะครอบครัว**

- **Gender** → `Gender_Male`, `Gender_Female`, `Gender_Other`
- **Marital Status** → `Marital_Single`, `Marital_Married`, `Marital_In_Relationship`, `Marital_Divorced`

### Scale Transformation: การแปลง Likert Scale และ Option Rating

**Likert Scale 5 ระดับ** → ใช้กับปัจจัยการตัดสินใจซื้อ 5 ข้อ + ปัจจัยบรรจุภัณฑ์ 8 ข้อ **(13 คอลัมน์)**

| ภาษาไทย | ค่าตัวเลข |
|--------|---------|
| มากที่สุด | 5 |
| มาก | 4 |
| ปานกลาง | 3 |
| น้อย | 2 |
| น้อยที่สุด | 1 |

**Option Rating 4 ระดับ** → ใช้กับ Option 1–10 ทั้ง 5 มิติ **(50 คอลัมน์)**

| ภาษาไทย | ค่าตัวเลข |
|--------|---------|
| เห็นด้วยที่สุด / เห็นด้วยอย่างยิ่ง | 4 |
| เห็นด้วย | 3 |
| เฉยๆ | 2 |
| ไม่เห็นด้วย / ไม่เห็นด้วยเลย | 1 |

### Feature Selection: การเลือกตัวแปรที่มีผลต่อ Target มากที่สุด

ใช้ **ANOVA F-test** เพื่อทดสอบว่าค่าเฉลี่ยของแต่ละ Feature แตกต่างกันระหว่างกลุ่ม Target Options หรือไม่

**หลักการ:**
- F-statistic สูง = Feature มีความแตกต่างระหว่างกลุ่มมาก
- p-value ต่ำ = ความแตกต่างนั้นมีนัยสำคัญทางสถิติ (p < 0.05)
- เลือก **Top 10 Features** ที่มี p-value ต่ำที่สุด

**Top 10 Features จาก ANOVA (เรียงตาม p-value ต่ำสุด):**

| Rank | Feature | F-statistic | p-value |
|------|---------|------------|---------|
| 1 | **Opt1_Premium_Feel** | 13.51 | 4.65e-12 |
| 2 | **Opt1_Purchase_Intent** | 12.18 | 5.45e-11 |
| 3 | **Opt1_Modernity** | 11.23 | 3.22e-10 |
| 4 | **Opt1_Attractiveness** | 9.29 | 1.45e-08 |
| 5 | Opt2_Purchase_Intent | 5.78 | 2.13e-05 |
| 6 | Opt2_Attractiveness | 5.32 | 5.64e-05 |
| 7 | Opt2_Modernity | 5.29 | 6.08e-05 |
| 8 | Opt2_Premium_Feel | 4.58 | 2.85e-04 |
| 9 | Opt3_Trust | 4.49 | 3.45e-04 |
| 10 | Opt1_Trust | 3.86 | 1.34e-03 |

### Random Forest Feature Importance

หลังจาก ANOVA คัด Top 10 แล้ว pipeline ยังใช้ **Random Forest** เพื่อวัด Feature Importance อีกชั้นบน **70 features ทั้งหมด** (Likert 13 + Option 50 + Age + Gender dummies 3 + Marital dummies 4):

**Top 10 Feature Importance (Random Forest):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 🏆 | **Factor_Ingredients** | 0.0763 |
| 2 | **Age_Ordinal** | 0.0760 |
| 3 | PkgFactor_Convenience | 0.0680 |
| 4 | PkgFactor_Promotion | 0.0640 |
| 5 | PkgFactor_Freshness | 0.0629 |
| 6 | Factor_Brand_Rep | 0.0597 |
| 7 | Factor_Price | 0.0584 |
| 8 | Factor_Taste | 0.0572 |
| 9 | PkgFactor_Price | 0.0565 |
| 10 | Factor_Packaging | 0.0543 |

> **Insight:** **Factor_Ingredients และ Age_Ordinal** มีอิทธิพลทำนายการเลือกบรรจุภัณฑ์ได้ดีที่สุด — ไม่ใช่ Option Rating แสดงว่าปัจจัยเรื่อง **ส่วนผสม/วัตถุดิบ** และ **อายุ** มีอิทธิพลมากกว่าการประเมินรูปลักษณ์บรรจุภัณฑ์

---

## Slide 6: Step 4.1 - Data Visualization (Demographics & Distribution)

### Demographic Profile — โปรไฟล์ผู้ตอบแบบสอบถาม

จากผู้ตอบ **148 คน** ที่ผ่านการ cleaning:

**การกระจายตามอายุ**

| ช่วงอายุ | จำนวน | สัดส่วน |
|---------|-------|--------|
| 30–39 ปี 🏆 | 62 | 41.9% |
| 20–29 ปี | 45 | 30.4% |
| 40–49 ปี | 27 | 18.2% |
| 50+ ปี | 14 | 9.5% |

> **Insight:** กลุ่มอายุ 20–39 ปี คิดเป็น **72.3%** ของผู้ตอบ — เป้าหมายหลักคือคนวัยทำงานและวัยรุ่นตอนปลาย

**การกระจายตามเพศ**

| เพศ | สัดส่วน |
|-----|--------|
| หญิง 🏆 | 73.0% |
| ชาย | 23.6% |
| อื่นๆ | 3.4% |

**การกระจายตามสถานะครอบครัว**

| สถานะ | จำนวน | สัดส่วน |
|-------|-------|--------|
| โสด ไม่มีแฟน (Single) | 52 | 35.1% |
| มีแฟนแต่ไม่แต่ง (In_Relationship) | 50 | 33.8% |
| แต่งงานแล้ว (Married) | 42 | 28.4% |
| หย่าร้าง/เป็นม่าย (Divorced) | 4 | 2.7% |

### Distribution Analysis — แนวโน้มความชอบที่พบจาก Text Mining

**คุณสมบัติที่ผู้บริโภคต้องการเห็นในบรรจุภัณฑ์ (จากการวิเคราะห์ข้อความปลายเปิด — พบจริง 46 คนจาก 148)**

| อันดับ | คุณสมบัติ | จำนวน | สัดส่วน |
|--------|---------|-------|--------|
| 1 🏆 | ตัวอักษรใหญ่/ชัดเจน (Need_Big_Text) | 11 | 7.4% |
| 2 | ของแถม/กิจกรรม interactive (Need_Interactive) | 10 | 6.8% |
| 3 | หน้าต่างใส (Need_Clear_Window) | 8 | 5.4% |
| 4 | ข้อมูลโซเดียม/สารอาหารชัดเจน (Need_Sodium_Info) | 7 | 4.7% |
| 5 | ฝาเทได้ (Need_Pour_Lid) | 7 | 4.7% |
| 6 | ดีไซน์การ์ตูน (Design_Cartoon) | 7 | 4.7% |

> **Insight:** ผู้บริโภคกลุ่มที่ใส่ใจสุขภาพต้องการ **ข้อมูลโภชนาการที่ชัดเจน** และ **ตัวอักษรขนาดใหญ่** (Need_Big_Text สูงสุด) ส่วน Design_Cartoon และ Need_Interactive สูงตามมา — แสดงว่าการออกแบบที่ดึงดูดและมีกิจกรรมโต้ตอบก็เป็นที่ต้องการ

> **Insight:** ผู้บริโภคไทยให้ความสำคัญกับ **ความสะดวกในการทดลองสินค้าใหม่** (ขนาดเล็กแบ่งซื้อ) และ **ดีไซน์ที่ดึงดูดความสนใจ** (การ์ตูน, ผิวด้าน) มากกว่าฟังก์ชันการใช้งานเพียงอย่างเดียว

---

## Slide 7: Step 4.2 - Correlation Heatmap (ความสัมพันธ์ของตัวแปร)

### ความสัมพันธ์ระหว่างปัจจัยต่างๆ กับ Target Variable

จาก ANOVA พบว่า **Top 10 Features** ที่มีผลต่อการเลือก Option มากที่สุด ล้วนเป็น **Option Attribute Scores** ได้แก่ Attractiveness, Trust, Modernity, Premium_Feel, Purchase_Intent ของแต่ละ Option

### ความสัมพันธ์เชิงบวกที่สำคัญ

| Feature Pair | ความหมาย |
|-------------|---------|
| Attractiveness ↔ Trust | บรรจุภัณฑ์ที่ดึงดูด = น่าเชื่อถือ |
| Attractiveness ↔ Premium_Feel | ความสวยงาม → ความรู้สึกพรีเมียม |
| Trust ↔ Purchase_Intent | ความเชื่อถือ → อยากซื้อ |
| Modernity ↔ Premium_Feel | ความทันสมัย → ความพรีเมียม |

> **Insight:** ผู้บริโภคมองว่า **ความน่าดึงดูด ความน่าเชื่อถือ และความพรีเมียม** เป็นสิ่งที่แยกไม่ออก — บรรจุภัณฑ์ที่ดีต้องมีทั้ง 3 อย่าง

### ความสัมพันธ์เชิงลบ / ที่น่าสนใจ

| Feature Pair | ความหมาย |
|-------------|---------|
| Need_Sodium_Info ↔ Option 3 | ผู้ที่ต้องการข้อมูลโซเดียมชัดเจน **ไม่ได้**เลือก Option 3 เป็นอันดับ 1 |
| Need_Clear_Window ↔ Design_Matte | ความต้องการเห็นสินค้าข้างใน vs ความชอบผิวด้าน (แยกกัน) |

> **Insight:** แบรนด์ต้องเลือกว่าจะเน้น **หน้าต่างใส** หรือ **ผิวด้านพรีเมียม** — ทั้งสองสไตล์ตรงข้ามกันในเชิงการออกแบบ

### Overall Mean Scores ของแต่ละ Option (1–4 Scale)

จากการประเมิน 10 Options × 5 Attributes (Attractiveness, Trust, Modernity, Premium_Feel, Purchase_Intent) บนมาตรวัด 1–4:

| อันดับ | Option | คะแนนเฉลี่ย |
|--------|--------|------------|
| 1 🏆 | **Option 3** | **2.56** |
| 2 | Option 1 | 2.54 |
| 3 | Option 2 | 2.51 |
| 4 | Option 6 | 2.49 |
| 5 | Option 7 | 2.47 |
| 6 | Option 4 | 2.38 |
| 7 | Option 5 | 2.36 |
| 8 | Option 9 | 2.31 |
| 9 | Option 8 | 2.30 |
| 10 | Option 10 | 2.23 |

> **หมายเหตุ:** แม้ Option 3 มีคะแนนเฉลี่ยสูงสุด แต่ ANOVA พบว่า **Opt1 มีความแตกต่างจากกลุ่มอื่นชัดเจนที่สุด** — นั่นหมายถึง Opt1 มี distinctive profile แม้ไม่ได้เป็นที่นิยมมากที่สุด

### Charts ที่สร้างจาก Pipeline (7 ภาพ)

| Chart | เนื้อหา |
|-------|---------|
| chart1_demographics.png | กราฟประชากรศาสตร์ (อายุ, เพศ, สถานะครอบครัว) |
| chart2_target_distribution.png | กราฟจำนวนเสียงแต่ละ Option (Opt3 นำโด่ง) |
| chart3_correlation_heatmap.png | Correlation ระหว่าง Top 10 ANOVA Features |
| chart4_mean_option_scores.png | คะแนนเฉลี่ยแต่ละ Option (1-4 scale) |
| chart5_insights_distribution.png | กระจายตัวของ 9 Insight Features |
| chart6_insights_vs_option.png | Insight Features แยกตาม Target Option |
| **chart7_rf_feature_importance.png** 🆕 | **Random Forest Feature Importance Top 10** |

---

## Slide 8: Conclusion & Strategic Recommendations (สรุปผลและข้อเสนอแนะ)

### สรุปความสมบูรณ์ของ Dataset

| ด้าน | สถานะ |
|------|-------|
| Missing Values | ✅ dropna → ลบ 19 แถวจาก 167 ที่มีข้อมูล |
| Incomplete Responses | ✅ กรองเฉพาะผู้มีประสบการณ์เลี้ยงแมว (Experience=เคย) |
| Open-ended Standardization | ✅ Breeds และ Brands จาก config.json |
| Encoding Consistency | ✅ Likert→1-5, Ordinal→1-4, One-Hot ถูกต้อง |
| Feature Selection | ✅ ANOVA F-test → Top 10 + RF Feature Importance |
| Charts | ✅ 7 charts |
| Ready for ML | ✅ Cleaned CSV (148 rows, 89 cols) พร้อมใช้งาน |

### Business Insights สำหรับแบรนด์ต่างประเทศ

**1. เน้นอัตลักษณ์ Option 1 (ประสบการณ์ที่แตกต่างชัดเจน)**
→ ANOVA พบว่า Opt1 มีความแตกต่างจาก Option อื่นๆ มากที่สุด (7 ใน 10 อันดับแรก) — บรรจุภัณฑ์ที่ดีต้องมี **ความน่าดึงดูด ความน่าเชื่อถือ และความทันสมัย** พร้อมกัน

**2. ปัจจัยส่วนผสมสำคัญที่สำรวจพบ (Factor_Ingredients)**
→ ANOVA พบว่า **ส่วนผสม/วัตถุดิบ** เป็นปัจจัยที่ผู้บริโภคให้ความสำคัญ (Likert 5-level) — แบรนด์ควรเน้นคุณภาพวัตถุดิบในการสื่อสาร

**3. กลุ่มเป้าหมายหลัก: อายุ 30–39 ปี และเพศหญิง**
→ กลุ่มอายุ 30–39 ปี คิดเป็น 41.9% และเพศหญิง 73% — ควรออกแบบการตลาดที่ดึงดูดกลุ่มนี้โดยเฉพาะ

**4. ข้อมูลโภชนาการและความชัดเจนของตัวอักษร**
→ Text mining พบว่าผู้บริโภคกลุ่มที่ใส่ใจสุขภาพต้องการข้อมูลโซเดียม/สารอาหาร (4.7%) และตัวอักษรขนาดใหญ่ (7.4%) — **Need_Big_Text** และ **Need_Sodium_Info** สูงสุดในกลุ่ม insight

**5. ดีไซน์การ์ตูนและกิจกรรมโต้ตอบ**
→ **Design_Cartoon (4.7%)** และ **Need_Interactive (6.8%)** สูง — การออกแบบที่น่ารักและมีของแถมดึงดูดความสนใจได้ดี

**6. ระวัง Generalization**
→ ข้อมูลมาจาก convenience sampling (n=148) และ Options 9, 10 ไม่มีผู้เลือกเลย — ผลวิเคราะห์ควรใช้เป็น **directional insight** เท่านั้น

### สิ่งที่ควรระวัง

- ข้อมูลมาจาก **convenience sampling** (n=148) — ไม่สามารถ generalize ไปประชากรทั้งหมดได้
- ใช้ **Top 1 choice** เท่านั้น — ลำดับที่ 2 และ 3 ถูกตัดทิ้ง
- ผลลัพธ์ควรใช้เป็น **directional insight** ไม่ใช่ definitive conclusion

---

## Slide 9: Q&A

**ขอบคุณครับ**

---

หากมีคำถามหรือต้องการข้อมูลเพิ่มเติม กรุณาติดต่อทีมวิจัย

**AIE323 Self-learning — Cat Food Packaging Preference Survey**
