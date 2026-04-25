# Supervised Model Recommendations
## AIE323 Data Pipeline — Packaging Choice Prediction

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 148 respondents |
| Features | 80 (Likert scales, demographics, option ratings) |
| Target Classes | 8 (Option 1-8) |
| Class Imbalance | Severe (Option 3 = 37.2%) |

---

## 🎯 Problem Type

**Multi-class Classification** — Predict which packaging option (1-10) a consumer will choose as their top preference.

---

## ✅ Recommended Models (Ranked)

### 1. **Random Forest Classifier** ⭐⭐⭐⭐⭐

**Why:**
- Handles imbalanced classes well with `class_weight='balanced'`
- Robust to overfitting with small dataset
- No feature scaling required
- Provides feature importance for interpretation

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Expected Performance:**
- Accuracy: 55-65%
- F1 (weighted): 0.50-0.60

---

### 2. **Gradient Boosting (XGBoost/LightGBM)** ⭐⭐⭐⭐

**Why:**
- Best for structured data
- Handles class imbalance with `scale_pos_weight`
- High accuracy on small datasets
- Provides probability outputs

```python
# XGBoost
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight='balanced',
    random_state=42,
    eval_metric='mlogloss'
)

# LightGBM (alternative)
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
```

**Expected Performance:**
- Accuracy: 58-68%
- F1 (weighted): 0.52-0.62

---

### 3. **Logistic Regression (Multinomial)** ⭐⭐⭐

**Why:**
- Interpretable coefficients
- Fast training
- Baseline model for comparison
- Good for probability calibration

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

**Expected Performance:**
- Accuracy: 48-58%
- F1 (weighted): 0.45-0.55

---

### 4. **SVM (RBF Kernel)** ⭐⭐⭐

**Why:**
- Works well with small datasets
- Effective in high-dimensional space
- Good for finding complex boundaries

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # For probability outputs
    random_state=42
)
```

**Expected Performance:**
- Accuracy: 50-60%
- F1 (weighted): 0.48-0.58

---

## ❌ Models to Avoid

| Model | Reason |
|-------|--------|
| Naive Bayes | Features are not independent |
| KNN | Too slow with 80 features, prone to overfitting |
| Decision Tree (single) | High variance, overfits easily |
| Neural Network | Too few samples (148) for training |

---

## 🔧 Preprocessing Steps

### 1. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 20-30 features
selector = SelectKBest(f_classif, k=25)
X_selected = selector.fit_transform(X, y)
```

### 2. Scaling (for LogReg, SVM, KNN)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Handle Imbalance
```python
# Option A: class_weight='balanced' (built into most models)

# Option B: SMOTE oversampling (if classes very imbalanced)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## 📈 Model Comparison Table

| Model | Accuracy | F1 (weighted) | Training Speed | Interpretability |
|-------|----------|--------------|----------------|------------------|
| **Random Forest** | 55-65% | 0.50-0.60 | Fast | Feature importance |
| **Gradient Boosting** | 58-68% | 0.52-0.62 | Medium | Feature importance |
| **Logistic Regression** | 48-58% | 0.45-0.55 | Fast | Coefficients |
| **SVM (RBF)** | 50-60% | 0.48-0.58 | Slow | Support vectors |

---

## 🎯 Validation Strategy

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Multiple metrics
scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    print(f"{model_name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 📋 Complete Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('cleaned_survey_data.csv')

# Prepare features
X = df.drop(columns=['Target_Option', 'Gender_Label', 'Marital_Label', 'Marital_Status'])
X = X.select_dtypes(include=[np.number]).fillna(X.median())

# Prepare target
le = LabelEncoder()
y = le.fit_transform(df['Target_Option'])

# Feature selection (top 25 features)
selector = SelectKBest(f_classif, k=25)
X_selected = selector.fit_transform(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test multiple models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial', class_weight='balanced', max_iter=1000
    ),
    'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True)
}

print("Model Performance (5-fold CV):")
print("="*50)
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
    print(f"{name:<25}: {scores.mean():.3f} ± {scores.std():.3f}")

# Train final model
best_model = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42
)
best_model.fit(X_scaled, y)

# Feature importance
feature_names = X.columns[selector.get_support()]
importances = best_model.feature_importances_
for feat, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
    print(f"\nTop Feature: {feat} = {imp:.4f}")
```

---

## 🎯 Recommended Approach

### Phase 1: Baseline
1. Start with **Random Forest** as baseline
2. Use 5-fold stratified CV
3. Evaluate with F1 (weighted) + Accuracy

### Phase 2: Improve
1. If F1 < 0.50 → Try **Gradient Boosting**
2. If overfitting → Add more regularization
3. Tune hyperparameters with GridSearchCV

### Phase 3: Production
1. Select best model
2. Train on full dataset
3. Save model + scaler + selector for prediction

---

## ⚠️ Important Warnings

1. **Small Dataset (148 samples)**
   - Cross-validation essential
   - Avoid overfitting
   - Simple models may outperform complex ones

2. **Class Imbalance**
   - Option 3 has 37% → model may bias toward Option 3
   - Use `class_weight='balanced'`
   - Evaluate with stratified splits

3. **Feature Space (80 features)**
   - High risk of overfitting
   - Use feature selection (top 20-30)
   - Consider PCA if features are correlated

---

## 📊 Expected Business Insights

From Random Forest feature importance, you can identify:

1. **Which factors drive packaging choice?**
   - Likert scores (Factor_* columns)
   - Demographics (Age, Gender)
   - Option-specific ratings (Opt*_*)

2. **Which demographic segments prefer which packaging?**
   - Age groups
   - Cat breed type (Purebred vs Mixed)
   - Current brand usage

3. **Recommendations for packaging design**
   - Optimize for factors with highest importance
   - Target specific demographics with tailored options

---

*Generated by AIE323 Pipeline*
*Date: April 2025*